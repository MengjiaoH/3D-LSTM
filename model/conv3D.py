import torch
import torch.nn as nn
import torch.nn.functional as F

class conv3d_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, opt=None):
        super().__init__()
        self.kernal_size = [2, 5, 5]

        self.conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size=self.kernal_size)

    def forward(self, x):
        out = self.conv_layer(x)
        ## this is in order to have the same effects as padding = same in tensorflow
        out = F.interpolate(self.conv_layer(x), size=x.shape[-3:], mode="nearest")
        
        return out