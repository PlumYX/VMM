import torch
import torch.nn as nn


class Model(nn.Module):
    ''' Linear Model '''
    def __init__(self, args):
        super(Model, self).__init__()

        self.linear = nn.Linear(args.in_dim, args.out_dim)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    class args(object):
        in_dim = 5
        out_dim = 3

    args = args()
    model = Model(args)
    print(model.parameters)

    a = torch.randn(2, 5)
    b = model(a).detach()
    print(a)
    print(b)
