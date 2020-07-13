import torch 
import torch.nn as nn
from model.e3d_lstm import E3DLSTM


class LSTM_Model(nn.Module):
    def __init__(self, opt=None):
        super(LSTM_Model, self).__init__()

        self.input_time_window = 4
        self.output_time_horizon = 1
        self.temporal_stride = 1
        self.temporal_frames = 2
        self.time_steps = (
            self.input_time_window - self.temporal_frames + 1
        ) // self.temporal_stride
        
        self.tau = 5
        self.hidden_size = 64
        self.lstm_layers = 4
        self.input_shape = (2, 4, 32, 32)
        self.output_shape = (2, 4, 32, 32)
        self.encoder = E3DLSTM(self.input_shape, self.hidden_size, self.lstm_layers, self.tau)
        self.decoder = nn.Conv3d(self.hidden_size * self.time_steps, output_shape[0], [2, 5, 5], padding=(0, 2, 2)) 

        