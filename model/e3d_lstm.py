import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import operator
from functools import reduce

from model.conv3D import conv3d_encoder
# CxT×H×W
# input_shape = (2, self.temporal_frames, 32, 32)
# output_shape = (2, self.output_time_horizon, 32, 32)

class E3DLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, tau):
        super().__init__()

        self.tau = tau
        self.cells = []

        input_shape = list(input_shape)
        # print("$ input shape", input_shape)
        
        for i in range(num_layers):
            cell = e3d_lstm_cell(input_shape, hidden_size)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self.cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # print("$ input size", input.size())
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []

        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self.cells):
                if step == 0:
                    c_history, m, h = self.cells[cell_idx].init_hidden(batch_size, self.tau, input.device)
                    c_history_states.append(c_history)
                    h_states.append(h)
                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history, m, h = cell(x, c_history_states[cell_idx], m, h_states[cell_idx])
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h

            outputs.append(h)

        # NOTE Concat along the channels
        # print("e3d lstm outputs size before cat", len(outputs), outputs[0].size())
        # print("e3d lstm outputs size after cat", len(outputs), outputs[0].size())
        return torch.cat(outputs, dim=1)

class e3d_lstm_cell(nn.Module):
    def __init__(self, input_shape, hidden_size, opt=None):
        super(e3d_lstm_cell, self).__init__()

        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.in_channel = input_shape[0]
        
        self.w_xi = conv3d_encoder(self.in_channel, self.hidden_size)
        self.w_hi = conv3d_encoder(self.hidden_size, self.hidden_size)

        self.w_xg = copy.deepcopy(self.w_xi)
        self.w_hg = copy.deepcopy(self.w_hi)

        self.w_xr = copy.deepcopy(self.w_xi)
        self.w_hr = copy.deepcopy(self.w_hi)

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.w_xi_prime = copy.deepcopy(self.w_xi)
        self.w_mi_prime = copy.deepcopy(self.w_hi)

        self.w_xg_prime = copy.deepcopy(self.w_xi)
        self.w_mg_prime = copy.deepcopy(self.w_hi)

        self.w_xf_prime = copy.deepcopy(self.w_xi)
        self.w_mf_prime = copy.deepcopy(self.w_hi)

        self.w_xo = copy.deepcopy(self.w_xi)
        self.w_ho = copy.deepcopy(self.w_hi)
        self.w_co = copy.deepcopy(self.w_hi)
        self.w_mo = copy.deepcopy(self.w_hi)

        self.w_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_attention(self, r, c_history):
        # Scaled Dot-Product but for tensors
        # instead of dot-product we do matrix contraction on twh dimensions
        scaling_factor = 1 / (reduce(operator.mul, r.shape[-3:], 1) ** 0.5)
        scores = torch.einsum("bctwh,lbctwh->bl", r, c_history) * scaling_factor

        attention = F.softmax(scores, dim=0)
        return torch.einsum("bl,lbctwh->bctwh", attention, c_history)
    
    def forward(self, x, c_history, m ,h):
        normalized_shape = list(h.shape[-3:])
        
        def LR(input):
            return F.layer_norm(input, normalized_shape)

        # R is CxT×H×W
        r = torch.sigmoid(LR(self.w_xr(x) + self.w_hr(h)))
        i = torch.sigmoid(LR(self.w_xi(x) + self.w_hi(h)))
        g = torch.tanh(LR(self.w_xg(x) + self.w_hg(h)))

        # TODO: what is self attention function
        recall = self.self_attention(r, c_history)

        c = i * g + self.layer_norm(c_history[-1] + recall)

        i_prime = torch.sigmoid(LR(self.w_xi_prime(x) + self.w_mi_prime(m)))
        g_prime = torch.tanh(LR(self.w_xg_prime(x) + self.w_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.w_xf_prime(x) + self.w_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(LR(self.w_xo(x) + self.w_ho(h) + self.w_co(c) + self.w_mo(m)))
        h = o * torch.tanh(self.w_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO first in first out?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)

        return (c_history, m, h)

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self.input_shape)
        memory_shape[0] = self.hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)

        return (c_history, m, h)

