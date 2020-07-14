import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.e3d_lstm import E3DLSTM
from utils import weights_init


class LSTM_Model(nn.Module):
    def __init__(self, opt=None):
        super(LSTM_Model, self).__init__()

        self.input_time_window = 4
        self.output_time_horizon = 4
        self.temporal_stride = 1
        self.temporal_frames = 2
        self.time_steps = (self.input_time_window - self.temporal_frames + 1) // self.temporal_stride
        # print("time steps ", self.time_steps)
        
        self.tau = 5
        self.hidden_size = 64
        self.lstm_layers = 4
        self.input_shape = (2, 2, 32, 32)
        self.output_shape = (2, 4, 32, 32)
        self.encoder = E3DLSTM(self.input_shape, self.hidden_size, self.lstm_layers, self.tau)
        self.decoder = nn.Conv3d(self.hidden_size * self.time_steps, self.output_shape[0], [1, 5, 5], padding=(1, 2, 2)) 

        # self.to(self.device)
        params = self.parameters(recurse=True)
        # TODO learning rate scheduler
        # Weight decay stands for L2 regularization
        self.optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0)

        self.apply(weights_init())
    
    def forward(self, x):
        out0 = self.encoder(x)
        out = self.decoder(out0)
        # print("@1 x ", x.size())
        # print("@2 out 0 ", out0.size())
        # print("@3 out ", out.size())
        return out
    
    def loss(self, x, y):
        # print("#1 out x shape", x.size())
        output = self(x)
        # print("#2 out x shape", output.size())

        l2_loss = F.mse_loss(output * 255, y * 255)
        l1_loss = F.l1_loss(output * 255, y * 255)

        return l1_loss, l2_loss