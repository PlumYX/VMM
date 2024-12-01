import torch
import torch.nn as nn

class Model(nn.Module):
    ''' Linear Model '''
    def __init__(self, args):
        super(Model, self).__init__()

        self.linear = nn.Linear(args.d_in, args.d_out)

    def forward(self, x):
        return self.linear(x)
