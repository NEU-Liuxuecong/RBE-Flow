from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.loss_config = config['rbe']['loss']
        # Preserve the existing configuration parameters.
        self.pos_w = self.loss_config.get('pos_weight', 1.0)
        self.neg_w = self.loss_config.get('neg_weight', 1.0)

    def compute_fine_matching_loss(self, data):
        """
        [Final Solution: Geometry-Aware Rectified Loss]
        Uses 's11_rect' to enforce Dynamic Trust Region.
        """
        gamma = 0.9  # Decay factor.
        
        if 'iter_flow_f' not in data or 'flow' not in data:
            return torch.tensor(0.0, device=data['image0'].device, requires_grad=True)
        
        flow_preds = data['iter_flow_f'] 
        flow_gt = data['flow']           
        cov_preds = data.get('iter_cov_f', None) 
        n_iters = len(flow_preds)
        
        total_loss = 0.0
        total_weight = 0.0
        
        # Key parameter: dynamic lower-bound coefficient.
        # 0.2 means the predicted standard deviation must cover at least 20% of the true error.
        alpha = 0.2 
        eps = 1e-6
        
        for i in range(n_iters):
            i_weight = gamma ** (n_iters - i - 1)
            total_weight += i_weight
            
            mu = flow_preds[i]
            
            # Residual (ground-truth residual).
            r = flow_gt - mu # [B, 2, H, W]
            
            # --- 1. Compute the base L1 loss (fallback) ---
            # If the Bayesian term becomes unstable, base_l1 keeps training grounded.
            base_l1 = torch.sqrt(torch.sum(r**2, dim=1) + eps)

            if cov_preds is not None:
                Sigma_raw = cov_preds[i]
                
                # Align spatial dimensions.
                if Sigma_raw.shape[-2:] != r.shape[-2:]:
                     B_s, _, _, H_s, W_s = Sigma_raw.shape
                     Sigma_raw = F.interpolate(Sigma_raw.view(B_s, 4, H_s, W_s), 
                                         size=r.shape[-2:], mode='nearest').view(B_s, 2, 2, r.shape[-2], r.shape[-1])
                
                # --- [Key] Implementation of s11_rect ---
                
                # 1. Take the squared ground-truth error (detach it and treat it as a constant base).
                r_detach = r.detach()

                # 2. Compute per-axis squared errors and dynamic floors separately
                # as specified in the paper: sigma_x^2 += alpha * error_x^2 ; sigma_y^2 += alpha * error_y^2
                r_sq_x = r_detach[:, 0]**2 + eps
                r_sq_y = r_detach[:, 1]**2 + eps
                min_variance_x = alpha * r_sq_x
                min_variance_y = alpha * r_sq_y

                # 3. Extract the raw predicted values.
                s11_raw = Sigma_raw[:, 0, 0]
                s22_raw = Sigma_raw[:, 1, 1]

                # 4. Apply rectification per-axis.
                # Softplus keeps the value positive, and per-axis min_variance keeps it above the true error.
                s11_rect = F.softplus(s11_raw) + min_variance_x
                s22_rect = F.softplus(s22_raw) + min_variance_y
                
                # --- NLL computation (using rectified values) ---
                
                # Determinant (for maximum stability, assume local independence and ignore off-diagonal terms).
                # This avoids NaNs caused by a non-positive-definite covariance matrix.
                det = s11_rect * s22_rect 
                log_det = torch.log(det + eps)
                
                # Mahalanobis
                # Even if r is large, this term will not explode because the denominator s11_rect is also large.
                inv_s11 = 1.0 / (s11_rect + eps) 
                inv_s22 = 1.0 / (s22_rect + eps)
                
                mahalanobis = inv_s11 * r[:,0]**2 + inv_s22 * r[:,1]**2
                
                # Pure Bayesian Loss
                nll_loss = 0.5 * (log_det + mahalanobis)
                
                # Final mix: L1 is primary and NLL is a 0.1 auxiliary term.
                # This is a safe formulation.
                loss_map = nll_loss#base_l1 + 0.1 * 

            # else:
            #     loss_map = base_l1

            # Masking
            # Mask handling: accept 'mask', or dataset-provided 'mask0'/'mask1'.
            if 'mask' in data:
                valid_mask = data['mask'].float().unsqueeze(1)
            elif 'mask0' in data and 'mask1' in data:
                # intersection of masks ensures both images valid
                valid_mask = (data['mask0'].float() & data['mask1'].float()).unsqueeze(1)
            elif 'mask0' in data:
                valid_mask = data['mask0'].float().unsqueeze(1)
            else:
                valid_mask = None

            if valid_mask is not None:
                if valid_mask.shape[-2:] != loss_map.shape[-2:]:
                    valid_mask = F.interpolate(valid_mask, size=loss_map.shape[-2:], mode='nearest')
                mask_2d = valid_mask.squeeze(1)
                loss_i = (loss_map * mask_2d).sum() / (mask_2d.sum() + 1e-6)
            else:
                loss_i = loss_map.mean()
            
            total_loss += i_weight * loss_i

        return total_loss / (total_weight + 1e-6)
    
    def compute_coarse_loss(self, data):
        """Coarse-level flow MAE loss."""
        if 'flow_c' in data:
            flow_pred = data['flow_c']
        elif 'flow_f_full' in data:
            flow_pred = data['flow_f_full']
            flow_pred = F.avg_pool2d(flow_pred, kernel_size=2, stride=2)
            # After downsampling predicted full-resolution flow, convert
            # values to coarse-grid pixel units by dividing by the scale (2).
            flow_pred = flow_pred.clone()
            flow_pred[:, 0, :, :] = flow_pred[:, 0, :, :] / 2.0
            flow_pred[:, 1, :, :] = flow_pred[:, 1, :, :] / 2.0
        else:
            return torch.tensor(0.0, device=data['image0'].device, requires_grad=True)
        
        flow_gt = data['flow']
        if flow_pred.shape != flow_gt.shape:
            # Compute vertical/horizontal scale factors separately to handle
            # non-square rescaling and ensure correct flow unit conversion.
            scale_h = flow_gt.shape[-2] // flow_pred.shape[-2]
            scale_w = flow_gt.shape[-1] // flow_pred.shape[-1]
            # Use (kh, kw) kernel to avg-pool spatially
            flow_gt_coarse = F.avg_pool2d(flow_gt, kernel_size=(scale_h, scale_w), stride=(scale_h, scale_w))
            # After downsampling the flow field, convert values from high-res
            # pixel units to low-res pixel units by dividing components by
            # the corresponding scale (width for u/x, height for v/y).
            flow_gt_coarse = flow_gt_coarse.clone()
            # channel 0 is horizontal (u / x), divide by horizontal scale
            flow_gt_coarse[:, 0, :, :] = flow_gt_coarse[:, 0, :, :] / float(scale_w)
            # channel 1 is vertical (v / y), divide by vertical scale
            flow_gt_coarse[:, 1, :, :] = flow_gt_coarse[:, 1, :, :] / float(scale_h)
        else:
            flow_gt_coarse = flow_gt
        
        flow_diff = torch.abs(flow_pred - flow_gt_coarse)
        
        if 'mask' in data:
            mask = data['mask'].float()
            # Downsample mask to prediction resolution using matching (h, w) scales
            scale_h = mask.shape[-2] // flow_pred.shape[-2]
            scale_w = mask.shape[-1] // flow_pred.shape[-1]
            if scale_h > 0 or scale_w > 0:
                mask = F.avg_pool2d(
                    mask.unsqueeze(1),
                    kernel_size=(scale_h or 1, scale_w or 1),
                    stride=(scale_h or 1, scale_w or 1)
                ).squeeze(1)
            mask = mask.unsqueeze(1)
            loss = (flow_diff * mask).sum() / (mask.sum() * 2 + 1e-8)
        else:
            loss = flow_diff.mean()
        
        return loss

    def forward(self, data):
        loss_scalars = {}
        
        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(data)
        loss_c = loss_c * self.loss_config.get('coarse_weight', 0.2)
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss (Kalman/Uncertainty Awareness is here)
        loss_f = self.compute_fine_matching_loss(data)
        loss_f = loss_f * self.loss_config.get('fine_weight', 1.0)
        loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})
        
        loss = loss_c + loss_f
        
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
        
        return loss
