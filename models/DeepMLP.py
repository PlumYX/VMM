import torch
import torch.nn as nn

class Monolayer_MLP(nn.Module):
    def __init__(self, d_in, d_out, layer_norm=True, act_func='GELU', dropout=.0):
        super(Monolayer_MLP, self).__init__()

        self.linear = nn.Linear(d_in, d_out)
        self.layer_norm = nn.LayerNorm(d_out) if layer_norm else None
        act_func_dict = {
            'ReLU': nn.ReLU(), 
            'ReLU6': nn.ReLU6(), 
            'LeakyReLU': nn.LeakyReLU(), 
            'GELU': nn.GELU(), 
            'Sigmoid': nn.Sigmoid(), 
            'Tanh': nn.Tanh()
        }
        self.act_func = act_func_dict[act_func]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.layer_norm:
            return self.dropout(self.act_func(self.layer_norm(self.linear(x))))
        else:
            return self.dropout(self.act_func(self.linear(x)))


class Model(nn.Module):
    ''' Deep multilayer perceptron '''
    def __init__(self, configs):
        super(Model, self).__init__()

        self.e_layers = configs.e_layers
        self.embedding_mapping = Monolayer_MLP(
            configs.d_in, configs.d_model, 
            configs.layer_norm, configs.act_func, configs.dropout
        )
        self.mlp_blacks = nn.ModuleList([
            Monolayer_MLP(
                configs.d_model, configs.d_model, 
                configs.layer_norm, configs.act_func, configs.dropout
            ) for _ in range(configs.e_layers-1)
        ]) if configs.e_layers != 1 else None
        self.projection = nn.Linear(configs.d_model, configs.d_out)

    def forward(self, x):
        x = self.embedding_mapping(x)
        if self.e_layers != 1: 
            for black in self.mlp_blacks:
                x = black(x)
        x = self.projection(x)
        return x
