import numpy as np
import torch
from torch.optim import SGD
from torch.nn.functional import cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch_cluster import random_walk
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch.utils.tensorboard import SummaryWriter

from concurrent.futures import ThreadPoolExecutor

from model import Model

INPUTS = 16
CHANNELS = 8
DIMENSIONS = 32
LAYERS = 12
SAMPLE_WALKS = 10000
LR = 5e-3
MOMENTUM = 0.95
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
        np.fromfile('tptp-graph/indices.dat', dtype=np.uint32)
            .astype(np.int64)
    )

    print("loading names...")
    problems = open('tptp-graph/problems.dat').read().split('\n')

    graph = Data(x=x, edge_index=edge_index)
    print("done")
    return graph, problem_index, problems

def sample(graph, problem_index):
    start = problem_index.repeat(SAMPLE_WALKS)
    walks = random_walk(
        graph.edge_index[0],
        graph.edge_index[1],
        start,
        walk_length=2 * LAYERS
    )
    visited = torch.unique(walks)
    x = graph.x[visited]
    edge_index, _ = subgraph(
        visited,
        graph.edge_index,
        relabel_nodes=True,
        num_nodes=len(graph.x)
    )
    data = Data(x=x, edge_index=edge_index)
    return data

def train():
    graph, problem_index, problems = load()

    model = Model(
        inputs=INPUTS,
        dimensions=DIMENSIONS,
        channels=CHANNELS,
        layers=LAYERS
    ).to('cuda')
    optimizer = SGD(
        model.parameters(),
        lr=LR/BATCH,
        nesterov=True,
        momentum=MOMENTUM
    )
    writer = SummaryWriter()

    example = 1
    checkpoint = 1
    while True:
        epoch_means = []
        epoch_labels = []

        shuffled = torch.randperm(len(problem_index))
        for step, (problem, data) in enumerate(
            ThreadPoolExecutor(1).map(
                lambda index: (
                    problems[index],
                    sample(graph, problem_index[index])
                ),
                shuffled
            )
        ):
            if step != 0 and step % BATCH == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(data)
            try:
                data = data.to('cuda')
                x = data.x
                edge_index = data.edge_index

                mean, variance, y = model(x, edge_index)
                reconstruction_loss = cross_entropy(y, x)
                divergence_loss = -0.5 * torch.sum(
                    1 + torch.log(variance) - mean.pow(2) - variance
                )
                loss = reconstruction_loss + divergence_loss
                loss.backward()
            except RuntimeError:
                print("OOM")
                continue

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
            epoch_means.append(mean.detach().to('cpu'))
            epoch_labels.append([problem, problem[:3]])

        epoch_means = torch.stack(epoch_means)
        writer.add_embedding(
            epoch_means,
            metadata=epoch_labels,
            metadata_header=['label', 'domain'],
            global_step=checkpoint
        )
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                writer.add_histogram(
                    name.replace('.', '/'),
                    parameter.data,
                    checkpoint
                )
        torch.save(model.state_dict(), 'model.pt')
        checkpoint += 1

if __name__ == '__main__':
    train()
