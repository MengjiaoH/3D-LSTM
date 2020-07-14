import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SlidingWindowDataset(Dataset):
    def __init__(self, data, transform=None, train=True, in_seq_len=4, out_seq_len=4, image_size=32, opt=None):
        self.train = train
        self.transform = transform
        self.raw_data = data
        self.image_size = image_size
        self.channel = 2
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        # if self.transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         ])

        num_train = len(data) * 2 // 3
        if self.train:
            start = 0
            end = num_train
        else:
            start = num_train
            end = len(data)
        self.data = data[start : end]

    def __len__(self):
        return self.data.shape[0] - self.in_seq_len - self.out_seq_len + 1
    
    def __getitem__(self, index):
        x = self.data[index : index + self.in_seq_len]
        y = self.data[index + self.in_seq_len : index + self.in_seq_len + self.out_seq_len]
        # switching to PyTorch format C,D,H,W
        x = np.swapaxes(x, 0, 1)
        y = np.swapaxes(y, 0, 1)
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return (torch.from_numpy(x).type(torch.float), 
                torch.from_numpy(y).type(torch.float),)
