import numpy as np
from model.conv3D import conv3d_encoder
import torch

print("testing 3D conv encoder")
input = torch.randn(20, 64, 20, 16, 16)

print(input.size())

model = conv3d_encoder(64, 64)
output = model(input)