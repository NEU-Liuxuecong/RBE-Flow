import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
from einops.einops import rearrange
import torch.nn.functional as F
import torch
import cv2
from .metrics import compute_mae, compute_rmse, compute_sr
import matplotlib.colors  # Required for color-space handling.


def flow_to_color(flow, max_flow=None):
    """
    Convert optical flow to a color-coded RGB image, similar to GMFlow visualization.
    
    Args:
        flow: Optical flow in [H, W, 2] or [2, H, W] format.
        max_flow: Maximum flow magnitude used for normalization.
    
    Returns:
        color_image: RGB image in [H, W, 3] format.
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)  # [2, H, W] -> [H, W, 2]
    
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    if max_flow is None:
        max_flow = np.max(magnitude)
    
    # Convert to HSV color space.
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi) * 255  # Hue
    hsv[:, :, 1] = 255  # Saturation
    hsv[:, :, 2] = np.clip(magnitude / max_flow * 255, 0, 255)  # Value
    
    # Convert to RGB.
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def make_flow_visualization(img0, img1, flow_pred, flow_gt=None, text=None, dpi=75):
    """
    Create a flow visualization figure in a GMFlow-like style.
    
    Args:
        img0, img1: Input images in [H, W] or [H, W, 3] format.
        flow_pred: Predicted flow in [2, H, W] or [H, W, 2] format.
        flow_gt: Ground-truth flow in [2, H, W] or [H, W, 2] format, optional.
        text: Text to display on the figure.
        dpi: Figure DPI.
    
    Returns:
        matplotlib figure
    """
    if flow_gt is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=dpi)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
        axes = [axes]
    
    # Ensure both images have three channels.
    if len(img0.shape) == 2:
        img0 = np.stack([img0] * 3, axis=-1)
    if len(img1.shape) == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    
    # First row: input images and predicted flow.
    axes[0][0].imshow(img0.astype(np.uint8))
    axes[0][0].set_title('Image 0')
    axes[0][0].axis('off')
    
    axes[0][1].imshow(img1.astype(np.uint8))
    axes[0][1].set_title('Image 1')
    axes[0][1].axis('off')
    
    # Predicted flow visualization.
    flow_pred_color = flow_to_color(flow_pred)
    axes[0][2].imshow(flow_pred_color)
    axes[0][2].set_title('Predicted Flow')
    axes[0][2].axis('off')
    
    if flow_gt is not None:
        # Second row: ground truth flow, error map, and flow difference.
        flow_gt_color = flow_to_color(flow_gt)
        axes[1][0].imshow(flow_gt_color)
        axes[1][0].set_title('Ground Truth Flow')
        axes[1][0].axis('off')
        
        # Compute the flow error.
        if isinstance(flow_pred, torch.Tensor):
            flow_pred_np = flow_pred.cpu().numpy()
        else:
            flow_pred_np = flow_pred
            
        if isinstance(flow_gt, torch.Tensor):
            flow_gt_np = flow_gt.cpu().numpy()
        else:
            flow_gt_np = flow_gt
        
        # Ensure the shapes match.
        if flow_pred_np.shape != flow_gt_np.shape:
            if flow_pred_np.ndim == 3 and flow_pred_np.shape[0] == 2:
                flow_pred_np = flow_pred_np.transpose(1, 2, 0)
            if flow_gt_np.ndim == 3 and flow_gt_np.shape[0] == 2:
                flow_gt_np = flow_gt_np.transpose(1, 2, 0)
        
        error = np.sqrt(np.sum((flow_pred_np - flow_gt_np)**2, axis=-1))
        
        # Error heatmap.
        im = axes[1][1].imshow(error, cmap='hot', vmin=0, vmax=np.percentile(error, 95))
        axes[1][1].set_title('Flow Error (EPE)')
        axes[1][1].axis('off')
        plt.colorbar(im, ax=axes[1][1], fraction=0.046, pad=0.04)
        
        # Flow difference visualization.
        flow_diff = flow_pred_np - flow_gt_np
        flow_diff_color = flow_to_color(flow_diff, max_flow=np.percentile(np.sqrt(np.sum(flow_diff**2, axis=-1)), 95))
        axes[1][2].imshow(flow_diff_color)
        axes[1][2].set_title('Flow Difference')
        axes[1][2].axis('off')
    
    # Add text annotations.
    if text is not None:
        text_str = '\n'.join(text)
        fig.text(0.02, 0.98, text_str, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def process_image(img_tensor):
    min_val = img_tensor.min()
    max_val = img_tensor.max()
    
    # If the maximum value is > 1, assume the image is already in the 0-255 range.
    if max_val > 1:
        img_np = img_tensor.numpy().astype(np.uint8)
    else:
        # Normalize to the 0-255 range.
        if max_val > min_val:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        img_np = (img_tensor.numpy() * 255).round().astype(np.uint8)
    
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    
    if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):  # Grayscale image.
        if img_np.ndim == 3:
            img_np = img_np.squeeze(axis=2)
        return np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 3:  # RGB image.
        return img_np  # Return RGB directly without conversion.
    else:  
        # For other channel counts (e.g. 4), use the first channel as grayscale.
        img_gray = img_np[:, :, 0]
        return np.stack([img_gray] * 3, axis=-1)


def _make_evaluation_figure(data, b_id, alpha='dynamic', ret_dict=None):
    """
    Create an evaluation figure using matching lines to visualize flow or keypoints.
    """
    img0 = data['image0'][b_id].cpu()
    img1 = data['image1'][b_id].cpu()

    img0 = process_image(img0)
    img1 = process_image(img1)

    H_orig, W_orig, _ = img0.shape

    if 'flow_f_full' in data and data['flow_f_full'] is not None:
        H_proc, W_proc = data['flow_f_full'][b_id].shape[1:]
    else:
        H_proc, W_proc = data['image0_model_shape']

    scale_h = H_orig / H_proc
    scale_w = W_orig / W_proc
    # Concatenate the two images horizontally.
    combined_img = np.concatenate([img0, img1], axis=1)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.imshow(combined_img)
    ax.axis('off')

    pts0, pts1_pred, pts1_gt = None, None, None  # Track ground-truth correspondences as well.
    avg_epe = -1
    epe_list = []  # Store per-point EPE values.

    # Reuse the precomputed AEPE from ret_dict when available.
    if ret_dict and 'metrics' in ret_dict and 'AEPE' in ret_dict['metrics']:
        if b_id < len(ret_dict['metrics']['AEPE']):
            avg_epe = ret_dict['metrics']['AEPE'][b_id]

    # Prefer keypoints for visualization when available.
    # if 'mkpts0_f' in data and 'mkpts1_f' in data:
    #     pts0 = data['mkpts0_f'][b_id].cpu().numpy()
    #     pts1_pred = data['mkpts1_f'][b_id].cpu().numpy()
        
        # If ground-truth flow is available but EPE has not been computed yet, compute it now.
        if avg_epe < 0 and 'flow' in data:
            flow_gt = data['flow'][b_id].cpu() # [2, H, W]
            if flow_gt.shape[-2:] != (H, W):
                flow_gt = F.interpolate(flow_gt.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)
            
            pts0_int = pts0.astype(int)
            pts0_int[:, 0] = np.clip(pts0_int[:, 0], 0, W - 1)
            pts0_int[:, 1] = np.clip(pts0_int[:, 1], 0, H - 1)
            
            flow_gt_pts = flow_gt[:, pts0_int[:, 1], pts0_int[:, 0]].T.numpy() # [N, 2]
            pts1_gt = pts0 + flow_gt_pts
            
            epe = np.linalg.norm(pts1_pred - pts1_gt, axis=1)
            avg_epe = np.mean(epe)

    # Otherwise, if flow data is available, fall back to corner-based sampling.
    if 'flow_f_full' in data:
        flow_pred = data['flow_f_full'][b_id].cpu()  # [2, H, W]

        # Use Harris corners on the original image for denser visualization.
        img0_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)  # Use the original image.
        corners = cv2.goodFeaturesToTrack(
            img0_gray, maxCorners=5000, qualityLevel=0.0001, minDistance=0.5)  # Tune parameters for the original resolution.

        if corners is not None and len(corners) > 0:
            pts0 = np.squeeze(corners, axis=1)  # [N, 2] in original resolution
            # Map points to the processed resolution to sample flow.
            pts0_proc = pts0 / np.array([scale_w, scale_h])
            pts0_proc_int = pts0_proc.astype(int)
            pts0_proc_int[:, 0] = np.clip(pts0_proc_int[:, 0], 0, W_proc - 1)
            pts0_proc_int[:, 1] = np.clip(pts0_proc_int[:, 1], 0, H_proc - 1)
        else:  # If no corners are detected, fall back to sparse grid sampling on the original image.
            step = 3
            y_coords, x_coords = np.mgrid[step//2:H_orig:step, step//2:W_orig:step]
            pts0 = np.stack((x_coords.ravel(), y_coords.ravel()), axis=-1)
            pts0_proc = pts0 / np.array([scale_w, scale_h])
            pts0_proc_int = pts0_proc.astype(int)
            pts0_proc_int[:, 0] = np.clip(pts0_proc_int[:, 0], 0, W_proc - 1)
            pts0_proc_int[:, 1] = np.clip(pts0_proc_int[:, 1], 0, H_proc - 1)

        # Sample corresponding points from the predicted flow.
        flow_pred_pts = flow_pred[:, pts0_proc_int[:, 1], pts0_proc_int[:, 0]].T.numpy()
        flow_pred_pts_scaled = flow_pred_pts * np.array([scale_w, scale_h])
        pts1_pred = pts0 + flow_pred_pts_scaled
        
        # Compute EPE at the processed resolution.
        if 'flow' in data:
            flow_gt = data['flow'][b_id].cpu()  # [2, H_proc, W_proc]
            flow_gt_pts = flow_gt[:, pts0_proc_int[:, 1], pts0_proc_int[:, 0]].T.numpy()
            pts1_gt_proc = pts0_proc + flow_gt_pts  # Ground-truth correspondences at the processed resolution.
            pts1_pred_proc = pts0_proc + flow_pred_pts  # Predicted correspondences at the processed resolution.
            epe_list = np.linalg.norm(pts1_pred_proc - pts1_gt_proc, axis=1)  # Compute EPE at the processed resolution.
            avg_epe = np.mean(epe_list) if len(epe_list) > 0 else -1

    if pts0 is not None and pts1_pred is not None:
        # Draw matching lines and keep only points with small EPE.
        for i in range(len(pts0)):
            # If EPE is available, filter out large errors; otherwise draw all points.
            if len(epe_list) > i and epe_list[i] >= 1.5:
                continue  # Skip large-error points.
            
            pt0 = pts0[i]
            pt1 = pts1_pred[i]
            if not (0 <= pt0[0] < W_orig and 0 <= pt0[1] < H_orig):
                continue  
            if not (0 <= pt1[0] < W_orig and 0 <= pt1[1] < H_orig):
                continue  
            
            green_color = (0.0, 1.0, 0.0, 0.95)
            ax.add_artist(plt.Circle(pt0, radius=0.5, color=green_color, fill=False, linewidth=1))
            ax.add_artist(plt.Circle((pt1[0] + W_orig, pt1[1]), radius=0.9, color=green_color, fill=False, linewidth=1))
                        
            line = plt.Line2D((pt0[0], pt1[0] + W_orig), (pt0[1], pt1[1]),
                              linewidth=1, color=green_color, alpha=0.8)
            ax.add_artist(line)

        if avg_epe >= 0:
            text = f'AEPE = {avg_epe:.2f}px'
            ax.text(0.5, -0.05, text, ha='center', va='center', transform=ax.transAxes, fontsize=12)

    else:
        ax.text(0.5, 0.5, "No matching data available", 
                color='white', fontsize=14, ha='center', va='center',
                bbox=dict(facecolor='red', alpha=0.7))

    plt.tight_layout()

    plt.tight_layout()
    return fig


def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure for flow
    raise NotImplementedError("Confidence visualization for flow not implemented")


def make_matching_figures(data, config, mode='evaluation', ret_dict=None):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_RBE.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA, ret_dict=ret_dict)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def make_mae_figures(data):
    """ Make mae figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_RBE_Pretrain.
    Returns:
        figures (List[plt.figure])
    """
    
    scale = data['hw0_i'][0] // data['hw0_f'][0]
    W_f = data["W_f"]

    pred0, pred1 = data["pred0"], data["pred1"]
    target0, target1 = data["target0"], data["target1"]

    # replace masked regions with predictions
    target0[data['b_ids'][data["ids_image0"]], data['i_ids'][data["ids_image0"]]] = pred0[data["ids_image0"]]
    target1[data['b_ids'][data["ids_image1"]], data['j_ids'][data["ids_image1"]]] = pred1[data["ids_image1"]]

    # remove excess parts, since the 10x10 windows have overlaping regions
    target0 = rearrange(target0, 'n l (h w) (p q c) -> n c (h p) (w q) l', h=W_f, w=W_f, p=scale, q=scale, c=1)
    target1 = rearrange(target1, 'n l (h w) (p q c) -> n c (h p) (w q) l', h=W_f, w=W_f, p=scale, q=scale, c=1) 
    # target0[:,:,-scale:,:] = 0.0
    # target0[:,:,:,-scale:] = 0.0
    # target1[:,:,-scale:,:] = 0.0
    # target1[:,:,:,-scale:] = 0.0
    gap = scale //2
    target0[:,:,-gap:,:] = 0.0
    target0[:,:,:,-gap:] = 0.0
    target1[:,:,-gap:,:] = 0.0
    target1[:,:,:,-gap:] = 0.0
    target0[:,:,:gap,:] = 0.0
    target0[:,:,:,:gap] = 0.0
    target1[:,:,:gap,:] = 0.0
    target1[:,:,:,:gap] = 0.0
    target0 = rearrange(target0, 'n c (h p) (w q) l -> n (c h p w q) l', h=W_f, w=W_f, p=scale, q=scale, c=1)
    target1 = rearrange(target1, 'n c (h p) (w q) l -> n (c h p w q) l', h=W_f, w=W_f, p=scale, q=scale, c=1)

    # windows to image 
    kernel_size = [int(W_f*scale), int(W_f*scale)]
    padding = kernel_size[0]//2 -1 if kernel_size[0] % 2 == 0 else kernel_size[0]//2
    stride = data['hw0_i'][0] // data['hw0_c'][0]
    target0 = F.fold(target0, output_size=data["image0"].shape[2:], kernel_size=kernel_size, stride=stride, padding=padding)
    target1 = F.fold(target1, output_size=data["image1"].shape[2:], kernel_size=kernel_size, stride=stride, padding=padding)

    # add mean and std of original image for visualization
    if ("image0_norm" in data) and ("image1_norm" in data):
        target0 = target0 * data["image0_std"] + data["image0_mean"]
        target1 = target1 * data["image1_std"] + data["image1_mean"]
        masked_image0 = data["masked_image0"] * data["image0_std"].to("cpu") + data["image0_mean"].to("cpu")
        masked_image1 = data["masked_image1"] * data["image1_std"].to("cpu") + data["image1_mean"].to("cpu")
    else:
        masked_image0 = data["masked_image0"] 
        masked_image1 = data["masked_image1"] 

    figures = []
    # Create a list of these tensors
    image_groups = [[data["image0"], masked_image0, target0],
                     [data["image1"], masked_image1, target1]]

    # Iterate through the batches
    for batch_idx in range(image_groups[0][0].shape[0]):  # Assuming batch dimension is the first dimension
        fig, axs = plt.subplots(2, 3, figsize=(9, 6))  
        for i, image_tensors in enumerate(image_groups):
            for j, img_tensor in enumerate(image_tensors):
                img = img_tensor[batch_idx, 0, :, :].detach().cpu().numpy()  # Get the image data as a NumPy array
                axs[i,j].imshow(img, cmap='gray', vmin=0, vmax=1)  # Display the image in a subplot with correct colormap
                axs[i,j].axis('off')  # Turn off axis labels
        fig.tight_layout()
        figures.append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
