import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from torch.optim import SGD
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_
from torch_cluster import random_walk
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.utils import subgraph, to_undirected
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns

from concurrent.futures import ThreadPoolExecutor# as TPEBase
from queue import Queue

from model import Encoder, Decoder

INPUTS = 16
CHANNELS = 8
DIMENSIONS = 2
LAYERS = 16
SAMPLE_WALKS = 10000
LR = 5e-3
BATCH = 16

def load():
    print("loading nodes...")
    x = torch.tensor(
        np.fromfile('tptp-graph/nodes.dat', dtype=np.uint8).astype(np.int64)
    )

    print("loading edge indices...")
    from_index = np.fromfile('tptp-graph/from.dat', dtype=np.uint32)\
        .astype(np.int64)
    to_index = np.fromfile('tptp-graph/to.dat', dtype=np.uint32)\
        .astype(np.int64)
    edge_index = torch.tensor([from_index, to_index])

    print("loading problem indices...")
    problem_index = torch.tensor(
        np.fromfile('tptp-graph/problems.dat', dtype=np.uint32)
            .astype(np.int64)
    )

    print("loading domains...")
    domains = open('tptp-graph/domain.dat').read().split('\n')

    graph = Data(x=x, edge_index=edge_index)
    print("done")
    return graph, problem_index, domains

def sample(graph, problem_index):
    start = problem_index.repeat(SAMPLE_WALKS)
    walks = random_walk(
        graph.edge_index[0],
        graph.edge_index[1],
        start,
        walk_length=LAYERS
    )
    visited = torch.unique(walks)
    x = graph.x[visited]
    edge_index, _ = subgraph(
        visited,
        graph.edge_index,
        relabel_nodes=True,
        num_nodes=len(graph.x)
    )
    problem_index = (visited == problem_index).nonzero().squeeze()

    data = Data(x=x, edge_index=edge_index)
    data.problem_index = problem_index
    return data

def train():
    sns.set()
    graph, problem_index, domains = load()

    encoder = Encoder(
        inputs=INPUTS,
        outputs=DIMENSIONS,
        channels=CHANNELS,
        layers=LAYERS
    ).to('cuda')
    decoder = Decoder(
            inputs=DIMENSIONS,
            outputs=INPUTS,
            channels=CHANNELS,
            layers=LAYERS
    ).to('cuda')
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = SGD(parameters, lr=LR / BATCH)

    writer = SummaryWriter()
    epoch = 1
    example = 1
    while True:
        x_coords = []
        y_coords = []
        domain_coords = []
        shuffled = torch.randperm(len(problem_index))
        print(f"epoch {epoch}...")
        for step, (domain, data) in enumerate(
            ThreadPoolExecutor(1).map(
                lambda index: (
                    domains[index],
                    sample(graph, problem_index[index])
                ),
                shuffled
            )
        ):
            if step != 0 and step % BATCH == 0:
                clip_grad_norm_(parameters, BATCH * 1.0)
                optimizer.step()
                optimizer.zero_grad()

            data = data.to('cuda')
            x = data.x
            edge_index = data.edge_index

            mean, log_variance = encoder(x, edge_index)
            std = torch.exp(0.5 * log_variance)
            random = torch.randn_like(std)
            z = mean + random * std
            y = decoder(z.repeat(len(x), 1), edge_index)

            reconstruction_loss = cross_entropy(y, x)
            divergence_loss = -0.5 * torch.sum(
                1 + log_variance - mean.pow(2) - log_variance.exp()
            )
            loss = reconstruction_loss + divergence_loss
            loss.backward()

            writer.add_scalar(
                'loss/reconstruction',
                reconstruction_loss,
                example
            )
            writer.add_scalar(
                'loss/divergence',
                divergence_loss,
                example
            )
            example += 1

            x_coords.append(mean[0].item())
            y_coords.append(mean[1].item())
            domain_coords.append(domain)

        data = pandas.DataFrame({
            'x': x_coords,
            'y': y_coords,
            'hue': domain_coords,
        })
        sns.relplot(
            x='x',
            y='y',
            hue='hue',
            hue_order=sorted(set(domain_coords)),
            data=data
        )
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.gcf().set_size_inches(12, 10)
        plt.savefig(f"figs/epoch-{epoch}.png")
        plt.close()

        for name, parameter in encoder.named_parameters():
            if parameter.requires_grad:
                writer.add_histogram(
                    name.replace('.', '/'),
                    parameter.data,
                    epoch
                )
        epoch += 1

if __name__ == '__main__':
    train()
