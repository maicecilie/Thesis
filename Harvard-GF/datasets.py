import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RNFLT_Dataset_Pred_Tds(Dataset):
    def __init__(self, data_path, subset='train', modality_type='rnflt'):
        self.data_path = data_path
        self.subset = subset
        self.modality_type = modality_type
        self.file_list = sorted(os.listdir(data_path))
        
        # optional split
        n = len(self.file_list)
        if subset == 'train':
            self.file_list = self.file_list[:int(n*0.8)]
        elif subset == 'val':
            self.file_list = self.file_list[int(n*0.8):int(n*0.9)]
        else:
            self.file_list = self.file_list[int(n*0.9):]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.file_list[idx])
        npz = np.load(path)
        
        if self.modality_type == 'rnflt':
            x = npz['rnflt_follow'][np.newaxis, :, :].astype(np.float32)
        elif self.modality_type == 'residual':
            x = npz['residual'][np.newaxis, :, :].astype(np.float32)
        elif self.modality_type == 'combined':
            x1 = npz['rnflt_follow'][np.newaxis, :, :]
            x2 = npz['residual'][np.newaxis, :, :]
            x = np.concatenate((x1, x2), axis=0).astype(np.float32)
        
        y = npz['tds'].astype(np.float32) / 38.0
        return torch.tensor(x), torch.tensor(y)
