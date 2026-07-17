from math import log
from loguru import logger

import torch
import torch.nn.functional as F
from einops import repeat
from kornia.utils import create_meshgrid
from einops.einops import rearrange

##############  ↓  Coarse-Level supervision  ↓  ##############

@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt

@torch.no_grad()
def compute_supervision_coarse(data, config):
    """
    Compute coarse-level supervision for both optical-flow and keypoint matching modes.
    """
    device = data['image0'].device
    
    # Check whether we are in optical-flow mode.
    if 'flow_f_full' in data:
        # Optical-flow mode does not need the legacy coarse-level supervision.
        # Create minimal supervision tensors for compatibility.
        data.update({
            'conf_matrix_gt': torch.zeros(1, 1, 1, device=device),
            'spv_b_ids': torch.tensor([0], device=device),
            'spv_i_ids': torch.tensor([0], device=device),
            'spv_j_ids': torch.tensor([0], device=device)
        })
        return
    
    # Legacy logic for keypoint matching mode.
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['RBE']['RESOLUTION'][0] #8
    if 'scale0' in data:
        scale0 = scale * data['scale0'][:, None]
        scale1 = scale * data['scale1'][:, None]
    else:
        scale0 = scale
        scale1 = scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt1_i = scale1 * grid_pt1_c

    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])
    grid_pt0_c = grid_pt0_i/scale0
    grid_pt1_c = grid_pt1_i/scale1

    flow = data['flow'].permute(0, 2, 3, 1)  # [N, H, W, 2]flow/2 3,0 [-18.5201,  -3.1385],
    '''print('flow shape:',flow.shape)
    print('flow min:',flow.min(),'max:',flow.max())# 4 256 256'''
    flow_coarse = F.interpolate(data['flow'], size=(h0, w0), mode='bilinear', align_corners=True)
    flow_coarse = flow_coarse / scale
    flow_coarse = flow_coarse.permute(0, 2, 3, 1)  # [N, h0, w0, 2][-2.1614, -0.5402]
    #print('flow_coarse shape:',flow_coarse.shape)# 4 32 32 2
    #print('flow_coarse min:',flow_coarse.min(),'max:',flow_coarse.max())
    grid_pt0_c_reshape = grid_pt0_c.reshape(N, h0, w0, 2)# 4 32 32
    warped_pt1_c = grid_pt0_c_reshape + flow_coarse  # [N, h0, w0, 2]#29 26.8386 0.4598
    warped_pt1_c_flat = warped_pt1_c.reshape(N, h0*w0, 2)

    grid_pt1_c_reshape = grid_pt1_c.reshape(N, h1, w1, 2)
    warped_pt0_c = grid_pt1_c_reshape - flow_coarse  # [N, h0, w0, 2]1tou0
    warped_pt0_c_flat = warped_pt0_c.reshape(N, h1*w1, 2)

    
    warped_pt1_c_round = warped_pt1_c_flat.round().long()
    warped_pt0_c_round = warped_pt0_c_flat.round().long()
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) | (pt[..., 0] > w-1) | (pt[..., 1] < 0) | (pt[..., 1] > h-1)
    
    #warped_pt1_c_round[..., 0].clamp_(0, w0-1)
    #warped_pt1_c_round[..., 1].clamp_(0, h0-10
    nearest_index1 = warped_pt1_c_round[..., 0] + warped_pt1_c_round[..., 1] * w1  # [N, hw0]
    out_bound=out_bound_mask(warped_pt1_c_round, w1, h1)
    nearest_index1[out_bound]=0

    nearest_index0 = warped_pt0_c_round[..., 0] + warped_pt0_c_round[..., 1] * w0  # [N, hw0]
    out_bound=out_bound_mask(warped_pt0_c_round, w0, h0)
    nearest_index0[out_bound]=0
    
    arange_1 = torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    arange_1[nearest_index1 == 0] = 0# n hw 2
    arange_b = torch.arange(N, device=device).unsqueeze(1)#N hw hw

    arange_0 = torch.arange(h1*w1, device=device)[None].repeat(N, 1)
    arange_0[nearest_index0 == 0] = 0
    
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    conf_matrix_gt[arange_b, arange_1, nearest_index1] = 1
    conf_matrix_gt[arange_b, nearest_index0,arange_0 ] = 1
    conf_matrix_gt[:, 0, 0] = False
    b_ids, i_ids, j_ids = conf_matrix_gt.nonzero(as_tuple=True)
    data.update({'conf_matrix_gt': conf_matrix_gt})


    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)
    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

##############  ↓  Fine-Level supervision  ↓  ##############

def compute_supervision_fine(data, config):
    """
    Compute fine-level supervision for both optical-flow and keypoint matching modes.
    """
    device = data['image0'].device
    
    # Check whether we are in optical-flow mode.
    if 'flow_f_full' in data:
        # Optical-flow mode does not need the legacy fine-level supervision.
        # Use flow ground truth directly for supervision.
        data.update({"conf_matrix_f_gt": torch.zeros(1, 1, 1, device=device)})
        return
    
    # Legacy logic for keypoint matching mode.
    N, _, H0, W0 = data['image0'].shape
    N, _, H1, W1 = data['image1'].shape
    scale = config['RBE']['RESOLUTION'][1]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    scale_f_c = config['RBE']['RESOLUTION'][0] // config['RBE']['RESOLUTION'][1]
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    W_f = config['RBE']['FINE_WINDOW_SIZE']
    
    # Check whether keypoint data is available.
    if 'b_ids' not in data or 'i_ids' not in data or 'j_ids' not in data:
        # No keypoints available; create an empty supervision matrix.
        data.update({"conf_matrix_f_gt": torch.zeros(1, W_f*W_f, W_f*W_f, device=device)})
        return
        
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
    if len(b_ids) == 0:
        data.update({"conf_matrix_f_gt": torch.zeros(1, W_f*W_f, W_f*W_f, device=device)})
        return
    # meshgrid for fine window
    grid_pt0_c = create_meshgrid(h0, w0, False, device).repeat(N, 1, 1, 1)
    grid_pt0_i = scale0[:,None,...] * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).repeat(N, 1, 1, 1)#.reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1[:,None,...] * grid_pt1_c

    stride_f = data['hw0_f'][0] // data['hw0_c'][0]
    grid_pt0_i = rearrange(grid_pt0_i, 'n h w c -> n c h w')
    grid_pt0_i = F.unfold(grid_pt0_i, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
    grid_pt0_i = rearrange(grid_pt0_i, 'n (c ww) l -> n l ww c', ww=W_f**2)
    grid_pt0_i = grid_pt0_i[b_ids, i_ids]
    
    grid_pt1_i = rearrange(grid_pt1_i, 'n h w c -> n c h w')
    grid_pt1_i = F.unfold(grid_pt1_i, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
    grid_pt1_i = rearrange(grid_pt1_i, 'n (c ww) l -> n l ww c', ww=W_f**2)
    grid_pt1_i = grid_pt1_i[b_ids, j_ids]
    
    flow = data['flow']
    flow_fine = F.unfold(flow, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
    flow_fine = rearrange(flow_fine, 'n (c ww) l -> n l ww c', ww=W_f**2)
    flow_fine = flow_fine[b_ids, i_ids]

    scale_factor_fine = config['RBE']['RESOLUTION'][1]  # 2 for fine level
    flow_fine = flow_fine/scale_factor_fine #flow/2
    warped_pt1_f = grid_pt0_i/scale0[b_ids] + flow_fine  # [M, W_f**2, 2]

    flow_fine_backward = -flow_fine  
    warped_pt0_f = grid_pt1_i/scale1[b_ids]+ flow_fine_backward  # [M, W_f**2, 2]
     
    warped_pt1_f_round = warped_pt1_f.round().long()
    warped_pt0_f_round = warped_pt0_f.round().long()  # Image 1 location obtained by warping image 0 with flow.

    nearest_index1 = warped_pt1_f_round[..., 0] + warped_pt1_f_round[..., 1] * W_f  # [M, W_f**2] # Warped image-1 indices.
    nearest_index0 = warped_pt0_f_round[..., 0] + warped_pt0_f_round[..., 1] * W_f  # [M, W_f**2] 
    
    M = warped_pt0_f.shape[0]
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] > w-1) + (pt[..., 1] < 0) + (pt[..., 1] > h-1)
    nearest_index1[out_bound_mask(warped_pt1_f_round, W_f, W_f)] = 0
    nearest_index0[out_bound_mask(warped_pt0_f_round, W_f, W_f)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0) # b i j 
    correct_0to1 = loop_back == torch.arange(W_f*W_f, device=device)[None].repeat(M, 1)
    correct_0to1[:, 0] = False 
    
    conf_matrix_f_gt = torch.zeros(M, W_f*W_f, W_f*W_f, device=device)
    b_ids_fine, i_ids_fine = torch.where(correct_0to1 != 0)
    j_ids_fine = nearest_index1[b_ids_fine, i_ids_fine]
    conf_matrix_f_gt[b_ids_fine, i_ids_fine, j_ids_fine] = 1
    
    data.update({"conf_matrix_f_gt": conf_matrix_f_gt})

