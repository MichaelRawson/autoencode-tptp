import torch
from torch.nn import BatchNorm1d, Embedding, Linear, Module, ModuleList
from torch.nn.functional import relu
from torch_scatter import scatter_mean

NODE_TYPES = 13
CHANNELS = 32
DIMENSIONS = 16
LAYERS = 8
EPS = 1e-8

class Conv(Module):
    def __init__(self):
        super().__init__()
        self.bn = BatchNorm1d(CHANNELS)
        self.weight = Linear(CHANNELS, CHANNELS)
        torch.nn.init.xavier_normal_(self.weight.weight)

    def forward(self, x, sources, targets):
        x = scatter_mean(x[sources], targets, dim_size=x.shape[0], dim=0)
        x = self.bn(x)
        return relu(self.weight(x))

class BiConv(Module):
    def __init__(self):
        super().__init__()
        self.out = Conv()
        self.back = Conv()

    def forward(self, x, sources, targets):
        out = self.out(x, sources, targets)
        back = self.back(x, targets, sources)
        return out + back

class Encoder(Module):
    def __init__(self):
        super().__init__()
        self.embed = Embedding(NODE_TYPES, CHANNELS)
        self.conv = ModuleList([BiConv() for _ in range(LAYERS)])
        self.mean = Linear(CHANNELS, DIMENSIONS)
        self.variance = Linear(CHANNELS, DIMENSIONS)

    def forward(self, nodes, sources, targets):
        x = self.embed(nodes)
        for conv in self.conv:
            x += conv(x, sources, targets)

        x = torch.mean(x, dim=0)
        mean = self.mean(x)
        log_variance = self.variance(x)
        return mean, log_variance

class Decoder(Module):
    def __init__(self):
        super().__init__()
        self.input = Linear(DIMENSIONS, CHANNELS)
        self.conv = ModuleList([BiConv() for _ in range(LAYERS)])
        self.output = Linear(CHANNELS, NODE_TYPES)

    def forward(self, x, sources, targets):
        x = relu(self.input(x))
        for conv in self.conv:
            x += conv(x, sources, targets)

        x = self.output(relu(x))
        return x

class Model(Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, nodes, sources, targets):
        mean, log_variance = self.encoder(nodes, sources, targets)
        std = torch.exp(0.5 * log_variance)
        random = torch.randn_like(std)
        x = mean + random * std
        x = x.unsqueeze(dim=0).repeat(nodes.shape[0], 1)
        reconstruction = self.decoder(x, sources, targets)
        return mean, log_variance, reconstruction

    def encode(self, nodes, sources, targets):
        return self.encoder(nodes, sources, targets)[0]
