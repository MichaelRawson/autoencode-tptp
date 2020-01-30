import torch
from torch.nn import Embedding, Linear, Module, ModuleList
from torch_geometric.nn import GCNConv
from torch.utils.checkpoint import checkpoint

class ConvLayer(Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_in = GCNConv(
	    channels, channels,
            bias=False,
	    flow='source_to_target'
	)
        self.conv_out = GCNConv(
	    channels, channels,
            bias=False,
	    flow='target_to_source'
	)

    def forward(self, x, edge_index):
        in_x = self.conv_in(x, edge_index)
        out_x = self.conv_out(x, edge_index)
        x = torch.cat((in_x, out_x), dim=1)
        x = torch.relu(x)
        return x

class DenseBlock(Module):
    def __init__(self, channels=None, layers=None):
        super().__init__()
        self.linear = ModuleList([
            Linear(2 * channels * (layer + 1), channels)
            for layer in range(layers)
        ])
        self.conv = ModuleList([
            ConvLayer(channels)
            for _ in range(layers)
        ])

    def forward(self, x, edge_index):
        inputs = [x]
        for conv, linear in zip(self.conv, self.linear):
            combined = torch.cat(inputs, dim=1)
            x = linear(combined)
            x = conv(x, edge_index)
            inputs.append(x)

        return torch.cat(inputs, dim=1)

class Encoder(Module):
    def __init__(self, inputs=None, outputs=None, channels=None, layers=None):
        super().__init__()
        self.embed = Embedding(inputs, 2 * channels)
        self.dense = DenseBlock(channels=channels, layers=layers)
        self.mean = Linear(2 * channels * (layers + 1), outputs)
        self.variance = Linear(2 * channels * (layers + 1), outputs)

    def forward(self, x, edge_index):
        x = self.embed(x)
        x = self.dense(x, edge_index)
        x = torch.max(x, dim=0)[0]
        mean = self.mean(x)
        variance = self.variance(x)
        return (mean, variance)

class Decoder(Module):
    def __init__(self, inputs=None, outputs=None, channels=None, layers=None):
        super().__init__()
        self.input = Linear(inputs, 2 * channels, bias=False)
        self.dense = DenseBlock(channels=channels, layers=layers)
        self.output = Linear(2 * channels * (layers + 1), outputs)

    def forward(self, x, edge_index):
        x = self.input(x)
        x = self.dense(x, edge_index)
        return self.output(x)
