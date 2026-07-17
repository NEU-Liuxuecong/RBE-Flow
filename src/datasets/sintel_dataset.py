import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
import struct

def read_flo(file_path):
    """
    Read .flo file in Middlebury format
    """
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print(f'Magic number incorrect. Invalid .flo file: {file_path}')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands) -> (h, w, 2)
            flow = np.resize(data, (h, w, 2))
            return flow

class SintelDataset(Dataset):
    def __init__(self, root_dir, pass_name='clean', is_training=True):
        """
        Args:
            root_dir (string): Directory with all the images (e.g. /media/ubuntu/data4t/Dataset/Sintel)
            pass_name (string): 'clean' or 'final'
            is_training (bool): If True, apply data augmentation
        """
        self.root_dir = root_dir
        self.pass_name = pass_name
        self.is_training = is_training
        
        self.img_dir = os.path.join(root_dir, 'training', pass_name)
        self.flow_dir = os.path.join(root_dir, 'training', 'flow')
        self.occ_dir = os.path.join(root_dir, 'training', 'occlusions')
        self.inv_dir = os.path.join(root_dir, 'training', 'invalid')
        
        self.scenes = sorted(os.listdir(self.img_dir))
        self.samples = []
        
        for scene in self.scenes:
            scene_img_dir = os.path.join(self.img_dir, scene)
            images = sorted(glob.glob(os.path.join(scene_img_dir, '*.png')))
            
            # For each scene, frames are paired N and N+1
            for i in range(len(images) - 1):
                img1_path = images[i]
                img2_path = images[i+1]
                
                frame_name = os.path.basename(img1_path)
                frame_base = os.path.splitext(frame_name)[0]
                
                flow_path = os.path.join(self.flow_dir, scene, frame_base + '.flo')
                occ_path = os.path.join(self.occ_dir, scene, frame_name)
                inv_path = os.path.join(self.inv_dir, scene, frame_name)
                
                if os.path.exists(flow_path) and os.path.exists(occ_path) and os.path.exists(inv_path):
                    self.samples.append({
                        'img1': img1_path,
                        'img2': img2_path,
                        'flow': flow_path,
                        'occ': occ_path,
                        'inv': inv_path
                    })

        # Augmentation
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Read images
        img1 = Image.open(sample['img1']).convert('RGB')
        img2 = Image.open(sample['img2']).convert('RGB')
        
        # Read masks (grayscale)
        occ = Image.open(sample['occ']).convert('L')
        inv = Image.open(sample['inv']).convert('L')
        
        # Read flow
        flow_gt = read_flo(sample['flow']) # (H, W, 2)
        flow_gt = torch.from_numpy(flow_gt).permute(2, 0, 1) # (2, H, W)
        
        # Create valid mask: only valid when both occ and inv are 0
        occ_np = np.array(occ)
        inv_np = np.array(inv)
        valid_mask = ((occ_np == 0) & (inv_np == 0)).astype(np.float32)
        valid_mask = torch.from_numpy(valid_mask).unsqueeze(0) # (1, H, W)
        
        # First resize the longer side to a smaller default size to avoid OOM in dense fine matching.
        # Tune this value to match training settings or available GPU memory.
        max_edge = 64
        w, h = img1.size
        if w > max_edge or h > max_edge:
            scale = max_edge / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img1 = TF.resize(img1, (new_h, new_w))
            img2 = TF.resize(img2, (new_h, new_w))
            valid_mask = torch.nn.functional.interpolate(valid_mask.unsqueeze(0), size=(new_h, new_w), mode='nearest').squeeze(0)
            flow_gt = torch.nn.functional.interpolate(flow_gt.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
            flow_gt[0, :, :] *= (new_w / w)
            flow_gt[1, :, :] *= (new_h / h)
            w, h = new_w, new_h

        # Resize & padding: RBE pos_embed assumes the input sequence is a perfect square (H = W).
        # Therefore, we resize and then pad the images to a square shape.
        max_size = max(h, w)
        # Ensure max_size is divisible by 8, as required by RBE.
        if max_size % 8 != 0:
            max_size = max_size + (8 - max_size % 8)
            
        pad_w = max_size - w
        pad_h = max_size - h
        
        if pad_w > 0 or pad_h > 0:
            img1 = TF.pad(img1, (0, 0, pad_w, pad_h), fill=0) # left, top, right, bottom
            img2 = TF.pad(img2, (0, 0, pad_w, pad_h), fill=0)
            
            # Pad the mask and flow tensors as well.
            # valid_mask: (1, h, w)
            valid_mask = TF.pad(valid_mask, (0, 0, pad_w, pad_h), fill=0)
            # flow_gt: (2, h, w)
            flow_gt = TF.pad(flow_gt, (0, 0, pad_w, pad_h), fill=0)

        if self.is_training:
            # Color jitter only on images
            if random.random() < 0.5:
                # Apply same or different jitter? Typically different or same depending on tasks. Let's do same.
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                    self.color_jitter.get_params(self.color_jitter.brightness, self.color_jitter.contrast,
                                                 self.color_jitter.saturation, self.color_jitter.hue)
                
                img1 = TF.adjust_brightness(img1, brightness_factor)
                img1 = TF.adjust_contrast(img1, contrast_factor)
                img1 = TF.adjust_saturation(img1, saturation_factor)
                img1 = TF.adjust_hue(img1, hue_factor)
                
                img2 = TF.adjust_brightness(img2, brightness_factor)
                img2 = TF.adjust_contrast(img2, contrast_factor)
                img2 = TF.adjust_saturation(img2, saturation_factor)
                img2 = TF.adjust_hue(img2, hue_factor)

            # Random horizontal flip
            if random.random() < 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
                valid_mask = TF.hflip(valid_mask)
                flow_gt = TF.hflip(flow_gt)
                # Flip u component of flow_gt
                flow_gt[0, :, :] *= -1
                
            # Random vertical flip
            if random.random() < 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
                valid_mask = TF.vflip(valid_mask)
                flow_gt = TF.vflip(flow_gt)
                # Flip v component of flow_gt
                flow_gt[1, :, :] *= -1

        # Transform to tensor (scales to [0, 1])
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        
        # Convert RGB to grayscale (since RBE usually handles 1-channel images)
        # Using standard weights: 0.2989 R + 0.5870 G + 0.1140 B
        img1_gray = (img1[0:1] * 0.2989) + (img1[1:2] * 0.5870) + (img1[2:3] * 0.1140)
        img2_gray = (img2[0:1] * 0.2989) + (img2[1:2] * 0.5870) + (img2[2:3] * 0.1140)

        # Generate padding masks (True for image, False for padding regions)
        coarse_scale = 0.125
        h_c = int(max_size * coarse_scale)
        w_c = int(max_size * coarse_scale)
        mask0 = torch.zeros((h_c, w_c), dtype=torch.bool)
        mask1 = torch.zeros((h_c, w_c), dtype=torch.bool)
        
        # Valid region before pad
        valid_h_c = int(h * coarse_scale)
        valid_w_c = int(w * coarse_scale)
        mask0[:valid_h_c, :valid_w_c] = True
        mask1[:valid_h_c, :valid_w_c] = True

        # Build dictionary just like OSDataset output
        data = {
            'image0': img1_gray,          # Shape: (1, 1024, 1024)
            'image1': img2_gray,
            'image0_path': sample['img1'],
            'image1_path': sample['img2'],
            'flow': flow_gt,              # Shape: (2, 440, 1024)
            'dataset_name': 'SintelDataset',
            'pair_id': idx,
            'pair_names': (os.path.basename(sample['img1']), os.path.basename(sample['img2'])),
            'mask': valid_mask.squeeze(0),         # Original valid mask for loss. Shape: (440, 1024)
            'mask0': mask0,                        # Padding masks for Transformer coarse-level
            'mask1': mask1
        }

        return data