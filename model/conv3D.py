import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, opt):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels)
        # self.conv_transpose3d = nn.ConvTranspose3d(out_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        # print(self.conv3d(input).shape, input.shape)
        # return self.conv_transpose3d(self.conv3d(input))
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")
