import torch
from torch.nn import Embedding, Linear, Module, ModuleList
from torch_geometric.nn import GCNConv
from torch.cuda import empty_cache

class DenseBlock(Module):
    def __init__(self, channels=None, layers=None):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.linear_in = ModuleList([
            Linear((2 * layer + 1) * channels, channels)
            for layer in range(layers)
        ])
        self.linear_out = ModuleList([
            Linear((2 * layer + 1) * channels, channels)
            for layer in range(layers)
        ])
        self.conv_in = ModuleList([
            GCNConv(channels, channels)
            for _ in range(layers)
        ])
        self.conv_out = ModuleList([
            GCNConv(channels, channels)
            for _ in range(layers)
        ])

    def forward(self, x, edge_index, edge_index_t):
        xs = [x]
        for linear_in, linear_out, conv_in, conv_out in zip(
            self.linear_in,
            self.linear_out,
            self.conv_in,
            self.conv_out
        ):
            x = torch.cat(xs, dim=1)
            x_in = torch.relu(linear_in(x))
            x_in = torch.relu(conv_in(x_in, edge_index))
            xs.append(x_in)
            x_out = torch.relu(linear_out(x))
            x_out = torch.relu(conv_out(x_out, edge_index_t))
            xs.append(x_out)

        return torch.cat(xs, dim=1)

class Encoder(Module):
    def __init__(self, inputs=None, outputs=None, channels=None, layers=None):
        super().__init__()
        self.embed = Embedding(inputs, channels)
        self.dense = DenseBlock(channels=channels, layers=layers)
        self.hidden = Linear((2 * layers + 1) * channels, 128)
        self.mean = Linear(128, outputs)
        self.variance = Linear(128, outputs)

    def forward(self, x, edge_index, edge_index_t):
        x = self.embed(x)
        x = self.dense(x, edge_index, edge_index_t)
        x = torch.mean(x, dim=0)
        x = torch.relu(self.hidden(x))
        mean = 2 * torch.tanh(self.mean(x))
        variance = 2 * torch.sigmoid(self.variance(x))
        return (mean, variance)

class Decoder(Module):
    def __init__(self, inputs=None, outputs=None, channels=None, layers=None):
        super().__init__()
        self.input = Linear(inputs, 128)
        self.hidden = Linear(128, channels)
        self.dense = DenseBlock(channels=channels, layers=layers)
        self.output = Linear((2 * layers + 1) * channels, outputs)

    def forward(self, z, num_nodes, edge_index, edge_index_t):
        x = torch.relu(self.input(z))
        x = torch.relu(self.hidden(x))
        x = x.repeat(num_nodes, 1)
        x = self.dense(x, edge_index, edge_index_t)
        return self.output(x)

class Model(Module):
    def __init__(self, inputs=None, dimensions=None, channels=None, layers=None):
        super().__init__()
        self.encoder = Encoder(
            inputs=inputs,
            outputs=dimensions,
            channels=channels,
            layers=layers
        )
        self.decoder = Decoder(
            inputs=dimensions,
            outputs=inputs,
            channels=channels,
            layers=layers
        )

    def forward(self, x, edge_index):
        edge_index_t = torch.stack((edge_index[1], edge_index[0]))
        mean, variance = self.encoder(x, edge_index, edge_index_t)
        std = torch.sqrt(variance)
        random = torch.randn_like(std)
        z = mean + random * std
        y = self.decoder(z, len(x), edge_index, edge_index_t)
        return mean, variance, y
