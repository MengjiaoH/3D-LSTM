import numpy as np
from model.conv3D import conv3d_encoder
import torch

print("testing 3D conv encoder")
input = torch.randn(8, 20, 16, 16, 16)

out = input[:, 2]
print(out.size())

# input_shape = [2, 64, 20, 16, 16]
# print(list(input_shape))

# model = conv3d_encoder(64, 64)
# output = model(input)