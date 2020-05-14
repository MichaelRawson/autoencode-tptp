import torch
from torch.nn import Embedding, Linear, Module, ModuleList, Parameter
from torch.nn.init import xavier_normal_, zeros_
import torch.nn.functional as F

EPS = 1e-8

class Encoder(Module):
    def __init__(self, inputs=None, outputs=None, channels=None, layers=None):
        super().__init__()
        self.embed = Embedding(inputs, channels)
        self.transform = ModuleList([
            Linear(channels, channels)
            for _ in range(layers)
        ])
        self.mean = Linear(channels, outputs)
        self.variance = Linear(channels, outputs)

    def forward(self, nodes, matrix, problem_index):
        x = self.embed(nodes)
        for i in range(len(self.transform)):
            print(f"encoder layer: {i}")
            transformed = self.transform[i](x)
            convolved = matrix @ x
            x = convolved
            F.relu_(x)
            del transformed, convolved

        x = x[problem_index]
        mean = 2 * torch.tanh(self.mean(x))
        variance = 2 * torch.sigmoid(self.variance(x))
        return mean, variance

class Decoder(Module):
    def __init__(self, inputs=None, outputs=None, channels=None, layers=None):
        super().__init__()
        self.input = Linear(inputs, channels)
        self.transform = ModuleList([
            Linear(channels, channels)
            for _ in range(layers)
        ])
        self.output = Linear(channels, outputs)

    def forward(self, x, z, matrix, problem_index):
        x[problem_index] = self.input(z)
        for i in range(len(self.transform)):
            print(f"decoder layer: {i}")
            transformed = self.transform[i](x)
            convolved = matrix @ x
            x = convolved
            F.relu_(x)
            del transformed, convolved

        x = self.output(x)
        return x

class Model(Module):
    def __init__(self, inputs=None, dimensions=None, channels=None, layers=None):
        super().__init__()
        self.channels = channels
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

    def forward(self, nodes, matrix_forward, matrix_back, problem_index):
        mean, variance = self.encoder(nodes, matrix_back, problem_index)
        std = torch.sqrt(variance + EPS)
        random = torch.randn_like(std)
        z = mean + random * std
        x = torch.zeros(nodes.shape[0], self.channels)
        y = self.decoder(x, z, matrix_forward, problem_index)
        return mean, variance, y
