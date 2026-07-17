import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_megadepth_gray,read_flow

class OSDataset(Dataset):
    def __init__(self,
                 root_dir,
                 list_path,
                 mode='train',
                 img_resize=64,
                 df=8,
                 img_padding=True,
                 **kwargs):
        """
        Manage OSDataset with optical and SAR image pairs.
        
        Args:
            root_dir (str): OSDataset root directory.
            list_path (str): path to train_list.txt, val_list.txt, or test_list.txt
            mode (str): options are ['train', 'val', 'test']
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        
        # for training RBE
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        
        # load scene list
        with open(list_path, 'r') as f:
            self.pair_list=[line.strip() for line in f.readlines()]
        
        logger.info(f"OSDataset {mode} mode: {len(self.pair_list)} scenes loaded")

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pair_id=self.pair_list[idx]
        '''
        opt_path=osp.join(self.root_dir,self.mode,'warped','512_30Trans0.1Scale10Rot','image_pair',f'{pair_id}_opt.tif')
        sar_path=osp.join(self.root_dir,self.mode,'warped','512_30Trans0.1Scale10Rot','image_pair',f'{pair_id}_sar_warped.tif')
        flow_path=osp.join(self.root_dir,self.mode,'warped','512_30Trans0.1Scale10Rot','truth_flow',f'{pair_id}.flo')
        '''
        '''
        opt_path=osp.join(self.root_dir,self.mode,'warped','512_30Trans0.1Scale360Rot','image_pair',f'{pair_id}_opt.tif')
        sar_path=osp.join(self.root_dir,self.mode,'warped','512_30Trans0.1Scale360Rot','image_pair',f'{pair_id}_sar_warped.tif')
        flow_path=osp.join(self.root_dir,self.mode,'warped','512_30Trans0.1Scale360Rot','truth_flow',f'{pair_id}.flo')
        '''
        '''
        opt_path=osp.join(self.root_dir,self.mode,'warped','512_20Trans0.2Scale30Rot','image_pair',f'{pair_id}_opt.tif')
        sar_path=osp.join(self.root_dir,self.mode,'warped','512_20Trans0.2Scale30Rot','image_pair',f'{pair_id}_sar_warped.tif')
        flow_path=osp.join(self.root_dir,self.mode,'warped','512_20Trans0.2Scale30Rot','truth_flow',f'{pair_id}.flo')
        '''
        
        
        opt_path=osp.join(self.root_dir,self.mode,'warped','512_15Trans0.1Scale30Rot','image_pair',f'{pair_id}_opt.tif')
        sar_path=osp.join(self.root_dir,self.mode,'warped','512_15Trans0.1Scale30Rot','image_pair',f'{pair_id}_sar_warped.tif')
        flow_path=osp.join(self.root_dir,self.mode,'warped','512_15Trans0.1Scale30Rot','truth_flow',f'{pair_id}.flo')
        
        '''
        opt_path=osp.join(self.root_dir,self.mode,'warped','512_0Trans0Scale90Rot','image_pair',f'{pair_id}_opt.tif')
        sar_path=osp.join(self.root_dir,self.mode,'warped','512_0Trans0Scale90Rot','image_pair',f'{pair_id}_sar_warped.tif')
        flow_path=osp.join(self.root_dir,self.mode,'warped','512_0Trans0Scale90Rot','truth_flow',f'{pair_id}.flo')
        '''
        
        # read grayscale images and masks. (1, h, w) and (h, w)
        image0, mask0, scale0 = read_megadepth_gray(
            opt_path, self.img_resize, self.df, self.img_padding, augment_fn=None)
        image1, mask1, scale1 = read_megadepth_gray(
            sar_path, self.img_resize, self.df, self.img_padding, augment_fn=None)
        
        if osp.exists(flow_path):
            flow=read_flow(flow_path)
            #print('flow shape:',flow.shape)#2 512 512
            #print('flow min:',flow.min(),'max:',flow.max())
            if self.img_resize is not None:
                h,w=image0.shape[1:] #1 256 256
                H0,W0=flow.shape[1:]
                
                # Compute the scale factors.
                scale_y=h/H0# H0 512
                scale_x=w/W0 #512
                # Resize the flow first, then rescale it.
                flow_resized = torch.nn.functional.interpolate(flow[None],size=(h,w),mode='bilinear',align_corners=True)[0]
                flow_resized[0] = flow_resized[0] * scale_x
                flow_resized[1] = flow_resized[1] * scale_y
                flow = flow_resized
                #print('flow shape:',flow.shape)# 2 256 256
                #print('flow min:',flow.min(),'max:',flow.max())
        else:
            flow=None
                

        ''' # create dummy camera matrices (identity for aligned images)
        K_0 = torch.eye(3, dtype=torch.float)
        K_1 = torch.eye(3, dtype=torch.float)
        
        # create dummy relative poses (identity for perfectly aligned images)
        # This eliminates geometric constraints for datasets without known camera poses
        T_0to1 = torch.eye(4, dtype=torch.float)
        T_1to0 = torch.eye(4, dtype=torch.float)
        
        # create realistic depth maps for aligned images
        # Use varying depths to simulate a natural scene
        h, w = image0.shape[1], image0.shape[2]
        
        # Create depth maps with spatial variation (simulating a natural scene)
        # Depths range from 0.5 to 10.0 meters with smooth gradients
        y_coords, x_coords = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing='ij')
        
        # Create depth variation based on image content and spatial position
        # Add some randomness but keep it consistent for the same scene
        torch.manual_seed(idx)  # Consistent depths for the same scene
        base_depth = 2.0 + 3.0 * (y_coords * 0.5 + x_coords * 0.3)  # Gradient from 2-5m
        noise = 0.5 * torch.randn(h, w)  # Small random variations
        depth0 = torch.clamp(base_depth + noise, 0.1, 20.0)  # Clamp to reasonable range
        
        # For aligned images, use the same depth (since they're perfectly registered)
        depth1 = depth0.clone()'''

        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'image0_path': opt_path,
            'image1_path': sar_path,
            #'T_0to1': T_0to1,  # (4, 4)
            #'T_1to0': T_1to0,
            #'K0': K_0,  # (3, 3)
            #'K1': K_1,
            'flow':flow, #flow/2
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'OSDataset',
            'pair_id': idx,
            'pair_names': (f'{pair_id}_opt.tif',f'{pair_id}_sar_warped.tif'),
        }
        if flow is not None:
            data['flow']=flow
            '''print('flow shape:',flow.shape)
            print('flow min:',flow.min(),'max:',flow.max())'''
        # for RBE training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data 