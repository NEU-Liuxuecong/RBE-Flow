import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import math


class SquareRootUKF(nn.Module):
    """
    [Innovation 2: Square-Root Unscented Kalman Filter for Optical Flow]
    Optimization: Fully Vectorized Sigma Point Processing to max out GPU utilization.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = 2  # State dimension (u, v)
        
        # UKF Hyperparameters
        self.alpha = 0.1
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ukf = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Precompute weights
        self.register_buffer('Wm', self._compute_weights_mean())
        self.register_buffer('Wc', self._compute_weights_cov())
        
        # --- NARKF Components ---
        # 1. Process Noise Q Predictor
        self.process_noise_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1), 
            nn.Softplus()
        )
        
        # 2. Measurement Noise R Predictor
        self.observation_noise_net = nn.Sequential(
            nn.Conv2d(hidden_dim + 18, 64, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 3, 1), 
            nn.Softplus()
        )
        
        # 3. Robustness Weight Predictor
        self.robustness_net = nn.Sequential(
            nn.Conv2d(hidden_dim + 18, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Measurement Function h(x)
        self.measurement_net = nn.Sequential(
            nn.Conv2d(hidden_dim + 18 + 2, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, 3, padding=1)
        )
        
    def _compute_weights_mean(self):
        Wm = torch.zeros(2 * self.n + 1)
        Wm[0] = self.lambda_ukf / (self.n + self.lambda_ukf)
        Wm[1:] = 1.0 / (2 * (self.n + self.lambda_ukf))
        return Wm
    
    def _compute_weights_cov(self):
        Wc = torch.zeros(2 * self.n + 1)
        Wc[0] = self.lambda_ukf / (self.n + self.lambda_ukf) + (1 - self.alpha**2 + self.beta)
        Wc[1:] = 1.0 / (2 * (self.n + self.lambda_ukf))
        return Wc
    
    def predict_dynamic_Q(self, context):
        B, _, H, W = context.shape
        params = self.process_noise_net(context)
        q11, q22 = params[:, 0] * 0.01 + 1e-6, params[:, 1] * 0.01 + 1e-6
        q12 = (params[:, 2] - 0.5) * 0.001
        Q = torch.zeros(B, 2, 2, H, W, device=context.device)
        Q[:, 0, 0], Q[:, 1, 1] = q11, q22
        Q[:, 0, 1], Q[:, 1, 0] = q12, q12
        return Q

    def predict_dynamic_R(self, context, cost_L0, cost_L1):
        inp = torch.cat([context, cost_L0, cost_L1], dim=1)
        params = self.observation_noise_net(inp)
        r11, r22 = params[:, 0] * 0.1 + 1e-4, params[:, 1] * 0.1 + 1e-4
        r12 = (params[:, 2] - 0.5) * 0.01
        B, _, H, W = context.shape
        R = torch.zeros(B, 2, 2, H, W, device=context.device)
        R[:, 0, 0], R[:, 1, 1] = r11, r22
        R[:, 0, 1], R[:, 1, 0] = r12, r12
        return R

    def predict_robust_weight(self, context, cost_L0, cost_L1):
        inp = torch.cat([context, cost_L0, cost_L1], dim=1)
        return self.robustness_net(inp).unsqueeze(1)

    def _generate_sigma_points(self, x, S):
        # x: [B, 2, H, W], S: [B, 2, 2, H, W]
        B, _, H, W = x.shape
        scale = math.sqrt(self.n + self.lambda_ukf)
        S_scaled = S * scale
        
        # Vectorized generation (No Lists/LOOPS)
        x_expanded = x.unsqueeze(1) # [B, 1, 2, H, W]
        S_cols = S_scaled.permute(0, 2, 1, 3, 4) # [B, 2, 2, H, W] -> columns treat
        
        # [Mean, Mean+S_0, Mean+S_1, Mean-S_0, Mean-S_1]
        sigma_points = torch.cat([
            x_expanded,
            x_expanded + S_cols,
            x_expanded - S_cols
        ], dim=1) # [B, 5, 2, H, W]
        
        return sigma_points
    
    def _measurement_function_vectorized(self, sigma_points, cost_L0, cost_L1, context):
        """
        Vectorized Processing: Process all 5 sigma points in one batch pass.
        sigma_points: [B, 5, 2, H, W]
        """
        B, num_sig, _, H, W = sigma_points.shape
        
        # 1. Collapse Batch and Sigma dims -> [B*5, ...]
        sigma_flat = sigma_points.view(B * num_sig, 2, H, W)
        
        # 2. Expand context/cost to match
        ctx_rep = context.unsqueeze(1).expand(-1, num_sig, -1, -1, -1).reshape(B * num_sig, -1, H, W)
        c0_rep = cost_L0.unsqueeze(1).expand(-1, num_sig, -1, -1, -1).reshape(B * num_sig, -1, H, W)
        c1_rep = cost_L1.unsqueeze(1).expand(-1, num_sig, -1, -1, -1).reshape(B * num_sig, -1, H, W)
        
        # 3. Concatenate and pass through network ONCE
        h_input = torch.cat([ctx_rep, c0_rep, c1_rep, sigma_flat], dim=1)
        z_pred_flat = self.measurement_net(h_input) # [B*5, 2, H, W]
        
        # 4. Reshape back
        z_pred = z_pred_flat.view(B, num_sig, 2, H, W)
        return z_pred
    
    def _cholesky_2x2(self, P):
        p11 = torch.clamp(P[:, 0, 0], min=1e-8)
        s11 = torch.sqrt(p11)
        s21 = P[:, 1, 0] / (s11 + 1e-8)
        p22_res = torch.clamp(P[:, 1, 1] - s21**2, min=1e-8)
        s22 = torch.sqrt(p22_res)
        
        zeros = torch.zeros_like(s11)
        row1 = torch.stack([s11, zeros], dim=1)
        row2 = torch.stack([s21, s22], dim=1)
        return torch.stack([row1, row2], dim=2)
    
    def forward(self, x_prev, S_prev, cost_L0, cost_L1, context_features, observed_flow):
        B, _, H, W = x_prev.shape
        device = x_prev.device
        
        # 1. Time Update
        x_pred = x_prev 
        P_prev = torch.einsum('bijhw,bkjhw->bikhw', S_prev, S_prev)
        Q_matrix = self.predict_dynamic_Q(context_features)
        P_pred = P_prev + Q_matrix
        S_pred = self._cholesky_2x2(P_pred)
        
        # 2. Generate Sigma Points (Vectorized)
        sigma_points = self._generate_sigma_points(x_pred, S_pred) # [B, 5, 2, H, W]
        
        # 3. Measurement Update (Vectorized - NO LOOPS)
        z_sigma_points = self._measurement_function_vectorized(
            sigma_points, cost_L0, cost_L1, context_features
        ) # [B, 5, 2, H, W]
        
        # 4. Predicted Mean (Vectorized Weighted Sum)
        Wm = self.Wm.view(1, -1, 1, 1, 1) # [1, 5, 1, 1, 1]
        z_pred = torch.sum(Wm * z_sigma_points, dim=1) # [B, 2, H, W]
        
        # 5. Covariances (Vectorized)
        z_diff = z_sigma_points - z_pred.unsqueeze(1) # [B, 5, 2, H, W]
        x_diff = sigma_points - x_pred.unsqueeze(1)   # [B, 5, 2, H, W]
        
        # Robust Einsum Implementation: Flatten spatial dims to avoid broadcasting errors
        B, num_sig, dim_n, H, W = z_diff.shape
        
        # [B*H*W, 5, 2]
        z_diff_flat = z_diff.permute(0, 3, 4, 1, 2).reshape(-1, num_sig, dim_n) 
        x_diff_flat = x_diff.permute(0, 3, 4, 1, 2).reshape(-1, num_sig, dim_n)
        Wc_flat = self.Wc # [5]
        
        # Calculate Covariance Per-Pixel
        # s=sigma, n=batch*pixels, i,j=vec_dim
        P_zz_flat = torch.einsum('s,nsi,nsj->nij', Wc_flat, z_diff_flat, z_diff_flat)
        P_xz_flat = torch.einsum('s,nsi,nsj->nij', Wc_flat, x_diff_flat, z_diff_flat)
        
        # Reshape back: [B*H*W, 2, 2] -> [B, H, W, 2, 2] -> [B, 2, 2, H, W]
        P_zz = P_zz_flat.view(B, H, W, 2, 2).permute(0, 3, 4, 1, 2)
        P_xz = P_xz_flat.view(B, H, W, 2, 2).permute(0, 3, 4, 1, 2)
        
        # Add R (Robust)
        R_matrix = self.predict_dynamic_R(context_features, cost_L0, cost_L1)
        weight = self.predict_robust_weight(context_features, cost_L0, cost_L1)
        R_robust = R_matrix / (weight + 1e-6)
        
        # Explicit shape check/broadcast removed as shape is now guaranteed [B, 2, 2, H, W]
        P_zz = P_zz + R_robust
        
        # 6. Kalman Gain
        det_Pzz = P_zz[:, 0, 0] * P_zz[:, 1, 1] - P_zz[:, 0, 1] * P_zz[:, 1, 0]
        det_Pzz_safe = torch.sign(det_Pzz) * torch.clamp(torch.abs(det_Pzz), min=1e-8)
        
        P_zz_inv = torch.zeros_like(P_zz)
        P_zz_inv[:, 0, 0] = P_zz[:, 1, 1] / det_Pzz_safe
        P_zz_inv[:, 0, 1] = -P_zz[:, 0, 1] / det_Pzz_safe
        P_zz_inv[:, 1, 0] = -P_zz[:, 1, 0] / det_Pzz_safe
        P_zz_inv[:, 1, 1] = P_zz[:, 0, 0] / det_Pzz_safe
        
        K = torch.einsum('bijhw,bjkhw->bikhw', P_xz, P_zz_inv)
        
        # 7. Update
        innovation = observed_flow - z_pred
        x_post = x_pred + torch.einsum('bijhw,bjhw->bihw', K, innovation)
        
        # 8. Covariance Update
        KPzzKt = torch.einsum('bijhw,bjkhw->bikhw',
                              torch.einsum('bijhw,bjkhw->bikhw', K, P_zz),
                              K.transpose(1, 2))
        P_post = P_pred - KPzzKt
        
        # Stability
        P_post = 0.5 * (P_post + P_post.transpose(1, 2))
        P_post[:, 0, 0] += 1e-6
        P_post[:, 1, 1] += 1e-6
        S_post = self._cholesky_2x2(P_post)
        
        return x_post, S_post, K, innovation

    
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SpectralManifoldFusion(nn.Module):
    def __init__(self, corr_channels=9):
        super().__init__()
        self.manifold_projector = nn.Sequential(
            nn.Conv2d(corr_channels * 2 + 2, 64, 1), 
            nn.InstanceNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 2, 1) 
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(corr_channels, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, corr_channels, 1)
        )

    def forward(self, corr_L0_sharp, corr_L1_blur):
        p_sharp = F.softmax(corr_L0_sharp, dim=1)
        h_sharp = -torch.sum(p_sharp * torch.log(p_sharp + 1e-6), dim=1, keepdim=True)
        
        p_blur = F.softmax(corr_L1_blur, dim=1)
        h_blur = -torch.sum(p_blur * torch.log(p_blur + 1e-6), dim=1, keepdim=True)
        
        manifold_input = torch.cat([corr_L0_sharp, corr_L1_blur, h_sharp, h_blur], dim=1)
        mixing_logits = self.manifold_projector(manifold_input)
        mixing_weights = F.softmax(mixing_logits, dim=1) 
        
        w_sharp = mixing_weights[:, 0:1]
        w_blur = mixing_weights[:, 1:2]
        
        fused_cost = w_sharp * corr_L0_sharp + w_blur * corr_L1_blur
        spectral_uncertainty = w_sharp * h_sharp + w_blur * h_blur
        
        return self.out_proj(fused_cost), spectral_uncertainty

class DeepLMSolverBlock(nn.Module):
    """
    Fixed: Numerical instability in 2x2 inverse.
    """
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual_conv = nn.Conv2d(128, 2, 3, padding=1)
        nn.init.constant_(self.residual_conv.weight, 0)
        nn.init.constant_(self.residual_conv.bias, 0)
        
        self.corr_encoder = nn.Sequential(nn.Conv2d(18, 64, 3, padding=1), nn.ReLU()) 
        self.resid_encoder = nn.Sequential(nn.Conv2d(input_dim*2, 64, 3, padding=1), nn.ReLU())
        self.flow_encoder = nn.Sequential(nn.Conv2d(2, 64, 7, padding=3), nn.ReLU())
        
        self.measure_fusion = nn.Conv2d(64 + 64 + 64, 128, 3, padding=1)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=128 + input_dim)
        
        self.j_prox_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 4, 3, padding=1)
        )
        
        self.g_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, 3, padding=1)
        )
        
        self.lambda_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Softplus() 
        )

    def forward(self, h, corr_L0, corr_L1, diff_local, diff_global, current_flow, context_features, P_prev=None):
        B, _, H, W = current_flow.shape
        
        # 1. Feature Encoding
        corr_feat = self.corr_encoder(torch.cat([corr_L0, corr_L1], dim=1))
        flow_feat = self.flow_encoder(current_flow)
        resid_feat = self.resid_encoder(torch.cat([diff_local, diff_global], dim=1))
        
        cost_features = F.elu(self.measure_fusion(torch.cat([corr_feat, flow_feat, resid_feat], dim=1)))
        
        # 2. GRU Update
        gru_in = torch.cat([context_features, cost_features], dim=1)
        h = self.gru(h, gru_in)
        
        # 3. Parameter Prediction
        J_flat = self.j_prox_head(h)
        J_prox = J_flat.view(B, 2, 2, H, W)
        g = self.g_head(h)
        lam = self.lambda_head(h)
        
        # 4. Levenberg-Marquardt Update
        H_mat = torch.einsum('bkihw,bkjhw->bijhw', J_prox, J_prox)
        epsilon = 1e-6
        eye = torch.eye(2, device=h.device).view(1, 2, 2, 1, 1)
        H_mat = H_mat + epsilon * eye
        
        if P_prev is not None:
            trace_P = P_prev[:, 0, 0] + P_prev[:, 1, 1] + 1e-8
            trace_P = trace_P.unsqueeze(1)
            lam = lam + 0.1 * trace_P

        M = H_mat.clone()
        M[:, 0, 0] = M[:, 0, 0] + lam[:, 0] + 1e-6
        M[:, 1, 1] = M[:, 1, 1] + lam[:, 1] + 1e-6
        
        # Safe 2x2 Inversion
        A = M[:, 0, 0]
        B_val = M[:, 0, 1]
        C_val = M[:, 1, 0]
        D = M[:, 1, 1]
        
        # Use simple determinant with safe guard
        det = A * D - B_val * C_val
        # Preserve sign but ensure magnitude
        det_sign = torch.sign(det)
        det_safe = det_sign * torch.clamp(torch.abs(det), min=1e-8)
        
        inv_A = D / det_safe
        inv_B = -B_val / det_safe
        inv_C = -C_val / det_safe
        inv_D = A / det_safe
        
        neg_g_u = -g[:, 0]
        neg_g_v = -g[:, 1]
        
        delta_u = inv_A * neg_g_u + inv_B * neg_g_v
        delta_v = inv_C * neg_g_u + inv_D * neg_g_v
        
        delta_f_math = torch.stack([delta_u, delta_v], dim=1)
        delta_f_net = self.residual_conv(h)
        
        # Clamp neural residual for stability
        delta_f_net = torch.clamp(delta_f_net, -2.0, 2.0)
        
        delta_f = delta_f_math + delta_f_net

        flow_new = current_flow + delta_f
        
        P_new = torch.stack([
            torch.stack([inv_A, inv_B], dim=1),
            torch.stack([inv_C, inv_D], dim=1)
        ], dim=1) 

        return h, flow_new, P_new, delta_f

class LocalFlowRefinement(nn.Module):
    """
    Fixed: Observation logic conflict. UKF now sees 'Observed Flow' which is Flow_prev + Delta_LM.
    """
    def __init__(self, dim, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.hidden_dim = dim
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.spectral_fusion = SpectralManifoldFusion(corr_channels=9)
        self.lm_solver = DeepLMSolverBlock(input_dim=dim, hidden_dim=dim)
        self.ukf_filter = SquareRootUKF(hidden_dim=dim)
        
        self.init_S_predictor = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Softplus()
        )

    def encode_features(self, feat0, feat1):
        f0_sharp = self.feature_extractor(feat0)
        f1_sharp = self.feature_extractor(feat1)
        
        f0_blur = F.avg_pool2d(f0_sharp, kernel_size=2, stride=2)
        f1_blur = F.avg_pool2d(f1_sharp, kernel_size=2, stride=2)
        
        f0_ctx_blur = F.avg_pool2d(f0_sharp, kernel_size=3, stride=1, padding=1)
        f1_ctx_blur = F.avg_pool2d(f1_sharp, kernel_size=3, stride=1, padding=1)
        
        return f0_sharp, f1_sharp, f0_blur, f1_blur, f0_ctx_blur, f1_ctx_blur
    
    def compute_local_correlation(self, f0, f1, radius=1):
        B, C, H, W = f0.shape
        num_neighbors = (2 * radius + 1) ** 2
        f1_padded = F.pad(f1, (radius, radius, radius, radius), mode='replicate')
        f1_unfolded = F.unfold(f1_padded, kernel_size=2*radius+1, padding=0)
        f1_unfolded = f1_unfolded.view(B, C, num_neighbors, H, W)
        f0_expanded = f0.unsqueeze(2)
        corr = torch.sum(f0_expanded * f1_unfolded, dim=1) 
        corr = corr / (C ** 0.5)
        return corr
    
    def forward(self, f0_sharp, f1_sharp, f0_blur, f1_blur, f0_ctx, f1_ctx, flow_init, num_iterations=10):
        h = torch.tanh(f0_sharp)
    
        # Init S
        init_val = self.init_S_predictor(f0_sharp) + 1e-4
        s11 = init_val[:, 0]
        s22 = init_val[:, 1]
        zeros = torch.zeros_like(s11)
        row1 = torch.stack([s11, zeros], dim=1)
        row2 = torch.stack([zeros, s22], dim=1)
        S = torch.stack([row1, row2], dim=2)

        flow = flow_init
        iter_flow_list = []
        iter_S_list = []
        
        # Initial save
        iter_flow_list.append(flow)
        iter_S_list.append(S)

        for i in range(num_iterations):
            flow_before_iter = flow.clone()

            # 1. Features & Cost
            warped_f1_sharp = self.warp_features(f1_sharp, flow)
            corr_L0 = self.compute_local_correlation(f0_sharp, warped_f1_sharp, radius=1)
            
            flow_down = F.interpolate(flow, size=f0_blur.shape[-2:], mode='bilinear', align_corners=True) * 0.5
            warped_f1_down = self.warp_features(f1_blur, flow_down)
            corr_L1 = self.compute_local_correlation(f0_blur, warped_f1_down, radius=1)
            
            corr_L1_up = F.interpolate(corr_L1, size=f0_sharp.shape[-2:], mode='nearest')
                
            diff_local = f0_sharp - warped_f1_sharp
            warped_f1_ctx = self.warp_features(f1_ctx, flow)
            diff_global = f0_ctx - warped_f1_ctx
            
            fused_cost, _ = self.spectral_fusion(corr_L0, corr_L1_up)

            # 2. LM Step (Proposal)
            P_prev = torch.einsum('bijhw,bkjhw->bikhw', S, S)
            h, flow_proposal, _, delta_lm = self.lm_solver(
                h, fused_cost, corr_L1_up, diff_local, diff_global, flow, f0_sharp, P_prev=P_prev
            )
            
            # 3. UKF Step (Filtering)
            # CRITICAL FIX: Observation for UKF is "Where LM thinks we should go"
            # It should be an absolute coordinate, not a delta.
            observed_flow = flow_before_iter + delta_lm 
            
            flow, S, _, _ = self.ukf_filter(
                flow, S, fused_cost, corr_L1_up, h, observed_flow
            )
            
            # 4. Clamping
            W_w = flow.shape[-1]
            max_displacement = W_w // 2
            # Calculate delta relative to start of this iteration
            delta_iter = flow - flow_before_iter
            delta_iter = torch.clamp(delta_iter, -10.0, 10.0) # Limit step size to avoid exploding
            
            flow = flow_before_iter + delta_iter

            iter_flow_list.append(flow)
            iter_S_list.append(S)

        return flow, S, iter_flow_list, iter_S_list 
        
    def warp_features(self, feat, flow):
        B, C, H, W = feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device),
            torch.arange(W, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        warped_grid = grid + flow
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        return F.grid_sample(feat, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

class FlowConfidenceEstimator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Conv2d(dim * 2 + 2, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, feat0, feat1, flow):
        warped_feat1 = self.warp_features(feat1, flow)
        feat_concat = torch.cat([feat0, warped_feat1, flow], dim=1)
        confidence = self.confidence_net(feat_concat)
        return confidence
    
    def warp_features(self, feat, flow):
        B, C, H, W = feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device),
            torch.arange(W, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        warped_grid = grid + flow
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        return F.grid_sample(feat, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

class FineMatching(nn.Module):
    """
    Fixed: Data dimensions and covariance upsampling.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_f = config['resnet']['block_dims'][0]
        self.dim = dim_f
        
        self.num_iterations =7
        
        self.feat_proj = nn.Sequential(
            nn.Linear(dim_f, dim_f),
            nn.GELU(),
            nn.Linear(dim_f, dim_f)
        )
        
        self.flow_refinement = LocalFlowRefinement(dim_f)
        self.confidence_estimator = FlowConfidenceEstimator(dim_f)
        
        self.edge_aware_smoother = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, 3, padding=1)
        )
       
    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        # 0. Preparation
        feat_f0, feat_f1, is_windowed = self._prepare_features(feat_f0_unfold, feat_f1_unfold, data)
        
        # 1. Main Loop
        flow_field, confidence, iter_flow_field, iter_confidence, iter_cov_field = self._compute_windowed_flow(feat_f0, feat_f1, data) 
        
        # 2. Post Processing
        flow_field = self._edge_aware_smoothing(flow_field, confidence)
        flow_full = self._upsample_to_image_resolution(flow_field, data)
        
        processed_iter_flow = []
        processed_iter_cov = []
        
        current_iters = len(iter_flow_field) # Might be less or equal to self.num_iterations + 1
        
        for i in range(current_iters):
            # Process Flow
            f = self._edge_aware_smoothing(iter_flow_field[i], iter_confidence[i])
            f_up = self._upsample_to_image_resolution(f, data)
            processed_iter_flow.append(f_up)
            
            # Process Covariance
            cov = iter_cov_field[i] # [B, 2, 2, H_f, W_f] (already permuted in _compute_windowed)
            
            # Upsample logic: ensure cov matches flow resolution if needed for loss
            tgt_h, tgt_w = f_up.shape[-2:]
            curr_h, curr_w = cov.shape[-2:]
            
            if curr_h != tgt_h:
                B, _, _, H_c, W_c = cov.shape
                # Flatten 2x2 -> 4 channels
                cov_flat = cov.view(B, 4, H_c, W_c)
                cov_up_flat = F.interpolate(cov_flat, size=(tgt_h, tgt_w), mode='nearest')
                cov_up = cov_up_flat.view(B, 2, 2, tgt_h, tgt_w)
                processed_iter_cov.append(cov_up)
            else:
                processed_iter_cov.append(cov)

        data.update({
            'flow_f': flow_field,
            'flow_f_full': flow_full,
            'flow_confidence': confidence,
            'iter_flow_f': processed_iter_flow,
            'iter_cov_f': processed_iter_cov,
            'num_iterations': self.num_iterations
        })
     
    def _prepare_features(self, feat_f0_unfold, feat_f1_unfold, data):
        if len(feat_f0_unfold.shape) == 4:
            B, L, WW, C = feat_f0_unfold.shape
            feat_f0_flat = feat_f0_unfold.view(B, L * WW, C)
            feat_f1_flat = feat_f1_unfold.view(B, L * WW, C)
        elif len(feat_f0_unfold.shape) == 3:  
            feat_f0_flat = feat_f0_unfold
            feat_f1_flat = feat_f1_unfold
            B, L, C = feat_f0_flat.shape
            WW = 1 # Dummy
        else:
            raise ValueError(f"Unsupported Feature Dim: {feat_f0_unfold.shape}")
        
        hw_c = data.get('hw0_c', (8, 8))  
        if isinstance(hw_c, torch.Size):
            hw_c = (hw_c[0], hw_c[1])
        
        H_fine = hw_c[0]  
        W_fine = hw_c[1]  
        
        feat_f0_proj = self.feat_proj(feat_f0_flat)
        feat_f1_proj = self.feat_proj(feat_f1_flat)
        
        if len(feat_f0_unfold.shape) == 4:
            feat_f0_windowed = feat_f0_proj.view(B, L, WW, self.dim)
            feat_f1_windowed = feat_f1_proj.view(B, L, WW, self.dim)
            return feat_f0_windowed, feat_f1_windowed, True
        else:
            # Fallback for full image processing if needed
            feat_f0_aggregated = feat_f0_proj
            feat_f1_aggregated = feat_f1_proj
            feat_f0 = feat_f0_aggregated.view(B, H_fine, W_fine, self.dim).permute(0, 3, 1, 2)
            feat_f1 = feat_f1_aggregated.view(B, H_fine, W_fine, self.dim).permute(0, 3, 1, 2)
            
            data['hw0_f'] = (H_fine, W_fine)
            data['hw1_f'] = (H_fine, W_fine)
            return feat_f0, feat_f1, False
    
    def _compute_windowed_flow(self, feat_f0_windowed, feat_f1_windowed, data):
        """Windowed Computation with Robust Reshaping"""
        B, L, WW, C = feat_f0_windowed.shape
        W_w = int(WW ** 0.5)
        device = feat_f0_windowed.device
        
        hw_c = data.get('hw0_c', (8, 8))
        H_f, W_f = hw_c

        # Batch Preparation
        # Treat each window as a separate batch item
        coarse_flow = data['flow_c'].to(device) # [B, 2, H_c, W_c]
        coarse_flow_points = coarse_flow.permute(0, 2, 3, 1).reshape(B, L, 2)

        window_feat0_batch = feat_f0_windowed.view(B * L, WW, C)
        window_feat1_batch = feat_f1_windowed.view(B * L, WW, C)
        
        window_feat0_batch = window_feat0_batch.permute(0, 2, 1).view(B * L, C, W_w, W_w)
        window_feat1_batch = window_feat1_batch.permute(0, 2, 1).view(B * L, C, W_w, W_w)
        
        flow_init_batch = coarse_flow_points.view(B * L, 2)
        flow_init_batch = flow_init_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, W_w, W_w)

        # --- CORE PROCESSING ---
        res = self._compute_local_window_flow(window_feat0_batch, window_feat1_batch, flow_init_batch)
        center_flow_batch, center_confidence_batch, iter_center_flow_batch, iter_center_confidence_batch, iter_center_cov_batch = res
        
        # Reshaping back to [B, H_f, W_f]
        # center_flow_batch: [B*L, 2]
        flows = center_flow_batch.view(B, H_f, W_f, 2).permute(0, 3, 1, 2)
        confidences = center_confidence_batch.view(B, H_f, W_f, 1).permute(0, 3, 1, 2)
        
        iter_flow_f = []
        iter_conf_f = []
        iter_cov_f = []
        
        # Process Iterations
        # iter_center_flow_batch is a LIST of [B*L, 2]
        for idx, (i_flow, i_conf, i_cov) in enumerate(zip(iter_center_flow_batch, iter_center_confidence_batch, iter_center_cov_batch)):
            # Flow
            f_view = i_flow.view(B, H_f, W_f, 2).permute(0, 3, 1, 2)
            iter_flow_f.append(f_view)
            
            # Confidence
            c_view = i_conf.view(B, H_f, W_f, 1).permute(0, 3, 1, 2)
            iter_conf_f.append(c_view)
            
            # Covariance: i_cov is [B*L, 2, 2]
            # Reshape to [B, H, W, 2, 2] -> Permute to [B, 2, 2, H, W]
            cov_view = i_cov.view(B, H_f, W_f, 2, 2)
            cov_view = cov_view.permute(0, 3, 4, 1, 2)
            iter_cov_f.append(cov_view)
        
        data['hw0_f'] = (H_f, W_f)
        data['hw1_f'] = (H_f, W_f)
       
        return flows, confidences, iter_flow_f, iter_conf_f, iter_cov_f

    def _compute_local_window_flow(self, window_feat0_batch, window_feat1_batch, flow_init_batch):
        """
        Batch Optimized DEKF Local Window Search
        """
        B_total, C, W_w, W_w = window_feat0_batch.shape
        device = window_feat0_batch.device
        
        f0_s, f1_s, f0_b, f1_b, f0_ctx, f1_ctx = self.flow_refinement.encode_features(window_feat0_batch, window_feat1_batch)
        
        # Run the full refinement loop
        final_flow, final_S, iter_flow_list, iter_S_list = self.flow_refinement(
            f0_s, f1_s, f0_b, f1_b, f0_ctx, f1_ctx,
            flow_init_batch,
            self.num_iterations
        )
            
        iter_center_flow_list = []
        iter_center_confidence_list = []
        iter_center_cov_list = []
        
        batch_indices = torch.arange(B_total, device=device)
        
        # Iterate through history
        for i, (flow, S) in enumerate(zip(iter_flow_list, iter_S_list)):
            
            # Use confidence estimator to find the best pixel in the window 
            # (or just take center if simple)
            current_conf = self.confidence_estimator(window_feat0_batch, window_feat1_batch, flow)
            
            # Strategy: Always take Center Pixel for structural regularity?
            # Or take max confidence? taking max confidence is risky for gradients if argmax is not smooth.
            # Let's stick to the center pixel for the "Coarse-to-Fine" grid logic.
            # The "Attention" happened in the window update.
            
            max_y, max_x = W_w // 2, W_w // 2
            
            # Extract
            current_iter_center_flow = flow[batch_indices, :, max_y, max_x] # [B_total, 2]
            current_iter_center_confidence = current_conf[batch_indices, 0, max_y, max_x].unsqueeze(1)
            
            current_iter_center_S = S[batch_indices, :, :, max_y, max_x] # [B_total, 2, 2]
            current_iter_center_P = torch.einsum('bij,bkj->bik', current_iter_center_S, current_iter_center_S)
            
            iter_center_flow_list.append(current_iter_center_flow)
            iter_center_confidence_list.append(current_iter_center_confidence)
            iter_center_cov_list.append(current_iter_center_P)
        
        center_flow = iter_center_flow_list[-1]
        center_confidence = iter_center_confidence_list[-1]
        
        return center_flow, center_confidence, iter_center_flow_list, iter_center_confidence_list, iter_center_cov_list

    def _edge_aware_smoothing(self, flow, confidence):
        weighted_flow = flow * confidence
        smoothed_flow = self.edge_aware_smoother(weighted_flow)
        alpha = 0.7 
        final_flow = alpha * smoothed_flow + (1 - alpha) * flow
        return final_flow
    
    def _upsample_to_image_resolution(self, flow_field, data):
        hw_f = data.get('hw0_f', flow_field.shape[-2:])
        hw_i = data.get('hw0_i', (hw_f[0] * 8, hw_f[1] * 8))
        
        if hw_i[0] > hw_f[0]:
            scale_factor = float(hw_i[0]) / float(hw_f[0])
            flow_full = F.interpolate(flow_field, size=hw_i, mode='bilinear', align_corners=True)
            flow_full = flow_full * scale_factor
        else:
            flow_full = flow_field
        
        return flow_full
