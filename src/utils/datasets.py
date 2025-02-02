import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os

class HHDCDataset(Dataset):
    def __init__(
            self,
            path: str,
            dtype: torch.dtype = torch.float,
            normalize: bool = True,
            get_slice: bool = False
    ):
        self.path = path
        self.ids = os.listdir(self.path)
        self.dtype = dtype
        self.norm = normalize
        self.get_slice = get_slice
    
    def __getitem__(self, index):
        item = np.load(f'{self.path}/{self.ids[index]}')
        item = torch.from_numpy(item).to(self.dtype)
        # normalize values to range [0,1]
        if self.norm:
            i_max,_ = torch.max(item,dim=0)
            i_min,_ = torch.min(item,dim=0)
            item = (item - i_min)/(i_max - i_min - 1e-12)   # add 1e-12 term to avoid nan
        # Select single slice
        if self.get_slice:
            item = torch.permute(item, (2,0,1))
            i = np.random.randint(0,item.shape[0])
            item = item[i].unsqueeze(0)
        else:
            item = item.unsqueeze(0)

        return item

    def __len__(self):
        return len(self.ids)