import torch
import torch.nn as nn
import math

class Square_items(nn.Module):
    def __init__(self, in_features, out_features):
        super(Square_items, self).__init__()

        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x*2)


class Cross_items(nn.Module):
    def __init__(self, in_features, out_features, sampling=1):
        super(Cross_items, self).__init__()

        self.sampling = sampling
        self.sampling_len = math.ceil(in_features / sampling)
        num_cross = int(self.sampling_len * (self.sampling_len - 1) / 2)
        self.fc = nn.Linear(num_cross, out_features)

    def forward(self, x):
        x_sampling = x
        x_sampling = x_sampling.unsqueeze(2)
        cross_item = torch.einsum("blk, bsk -> bls", x_sampling, x_sampling)
        cross_index = torch.triu(
            torch.ones((self.sampling_len, self.sampling_len), dtype=torch.bool), 
            diagonal=1)
        cross_item = cross_item[:, cross_index]
        return self.fc(cross_item)


class Model(nn.Module):
    ''' Quadratic polynomial regression '''
    def __init__(self, configs):
        super(Model, self).__init__()

        self.linear = nn.Linear(configs.d_in, configs.d_out)
        self.cross = Cross_items(configs.d_in, configs.d_out)
        self.square_items = Square_items(configs.d_in, configs.d_out)

    def forward(self, x):
        return self.linear(x) + self.cross(x) + self.square_items(x)
