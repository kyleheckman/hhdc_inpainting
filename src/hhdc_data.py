import torch
from torch.utils.data import Dataset
import numpy as np
import os

class HHDCDataset(Dataset):
    def __init__(
            self,
            path: str,
            dtype: torch.dtype = torch.float
    ):
        self.path = path
        self.ids = os.listdir(self.path)
        self.dtype = dtype
    
    def __getitem__(self, index):
        item = np.load(f'{self.path}/{self.ids[index]}')
        item = torch.from_numpy(item)
        return item.to(self.dtype)

    def __len__(self):
        return len(self.ids)