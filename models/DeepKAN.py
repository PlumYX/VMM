import torch
import torch.nn as nn

from layers.KANLayer import KANLayer

class Monolayer_KAN(nn.Module):
    def __init__(self, d_in, d_out, layer_norm=True):
        super(Monolayer_KAN, self).__init__()

        self.kan_layer = KANLayer(d_in, d_out)
        self.layer_norm = nn.LayerNorm(d_out) if layer_norm else None

    def forward(self, x):
        x, _, _, _ = self.KanLayer(x)
        x = self.layer_norm(x) if self.layer_norm else x
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.e_layers = configs.e_layers
        self.embedding_mapping = Monolayer_KAN(
            configs.d_in, configs.d_model, configs.layer_norm)
        self.kan_blacks = nn.ModuleList([
            Monolayer_KAN(
                configs.d_model, configs.d_model, configs.layer_norm
                ) for _ in range(configs.e_layers-1)
            ]) if configs.e_layers != 1 else None
        self.projection = nn.Linear(configs.d_model, configs.d_out)

    def forward(self, x):
        x = self.embedding_mapping(x)
        if self.e_layers != 1: 
            for black in self.kan_blacks:
                x = black(x)
        x = self.projection(x)
        return x
