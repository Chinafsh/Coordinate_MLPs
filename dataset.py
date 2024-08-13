import torch
from torch.utils.data import Dataset
import numpy as np
import imageio
from einops import rearrange
from kornia import create_meshgrid
from typing import Tuple
import cv2

class ImageDataset(Dataset):
    def __init__(self,
                 image_path:str,
                 split: str):
        """_summary_

        Args:
            image_path (str):iamge path
            split (str):train or validation
        """
        image = imageio.imread(image_path)[..., :3]/255
        c = [image.shape[0]//2, image.shape[1]//2]
        self.r = 256
        image = image[c[0]-self.r:c[0]+self.r, c[1]-self.r:c[1]+self.r]
        image = cv2.resize(image, (256, 256))
        
        
        
        self.uv = create_meshgrid(self.r, self.r, True)[0]   # (r, r, 2)
        self.rgb = torch.FloatTensor(image)  # (256, 256, 3)

        if split=='train':
            self.uv = self.uv[::2, ::2] # (128, 128, 2)
            self.rgb = self.rgb[::2, ::2]   #(128, 128, 3)
            
        self.uv = rearrange(self.uv, 'h w c -> (h w) c')
        self.rgb = rearrange(self.rgb, 'h w c -> (h w) c')
        
        
        
    def __len__(self):
        return len(self.uv) # self.uv.shape[0]
    
    def __getitem__(self, idx:int):
        
        return {"uv":self.uv[idx], 
                "rgb" : self.rgb[idx]
                }