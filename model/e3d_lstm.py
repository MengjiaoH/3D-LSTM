import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import operator

from model.conv3D import conv3d_encoder
# CxT×H×W
# input_shape = (2, self.temporal_frames, 32, 32)
# output_shape = (2, self.output_time_horizon, 32, 32)

class E3DLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, tau):
        super().__init__()

        self._tau = tau
        self._cells = []

        input_shape = list(input_shape)
        
        for i in range(num_layers):
            cell = E3DLSTMCell(input_shape, hidden_size)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []

        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self._cells):
                if step == 0:
                    c_history, m, h = self._cells[cell_idx].init_hidden(
                        batch_size, self._tau, input.device
                    )
                    c_history_states.append(c_history)
                    h_states.append(h)

                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history, m, h = cell(
                    x, c_history_states[cell_idx], m, h_states[cell_idx]
                )
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h

            outputs.append(h)

        # NOTE Concat along the channels
        return torch.cat(outputs, dim=1)


class e3d_lstm_cell(nn.Module):
    def __init__(self, input_shape, hidden_size, opt=None):
        super(e3d_lstm_cell, self).__init__()

        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.in_channel = input_shape[0]
        
        self.w_xi = conv3d_encoder(self.in_channel, self.hidden_size)
        self.w_hi = conv3d_encoder(self.hidden_size, self.hidden_size)

        self.w_xg = copy.deepcopy(self.weight_xi)
        self.w_hg = copy.deepcopy(self.weight_hi)

        self.w_xr = copy.deepcopy(self.weight_xi)
        self.w_hr = copy.deepcopy(self.weight_hi)

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.w_xi_prime = copy.deepcopy(self.weight_xi)
        self.w_mi_prime = copy.deepcopy(self.weight_hi)

        self.w_xg_prime = copy.deepcopy(self.weight_xi)
        self.w_mg_prime = copy.deepcopy(self.weight_hi)

        self.w_xf_prime = copy.deepcopy(self.weight_xi)
        self.w_mf_prime = copy.deepcopy(self.weight_hi)

        self.w_xo = copy.deepcopy(self.weight_xi)
        self.w_ho = copy.deepcopy(self.weight_hi)
        self.w_co = copy.deepcopy(self.weight_hi)
        self.w_mo = copy.deepcopy(self.weight_hi)

        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_atttention(self, r, c_history):
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
        r = torch.sigmoid(LR(self.weight_xr(x) + self.weight_hr(h)))
        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))
        recall = self.self_atttention(r, c_history)

        c = i * g + self.layer_norm(c_history[-1] + recall)

        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(
            LR(
                self.weight_xo(x)
                + self.weight_ho(h)
                + self.weight_co(c)
                + self.weight_mo(m)
            )
        )
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO first in first out?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)

        return (c_history, m, h)

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)

        return (c_history, m, h)

