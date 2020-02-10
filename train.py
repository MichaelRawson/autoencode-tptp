import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from model import Model, EPS

INPUT = 16
CHANNELS = 8
DIMENSIONS = 8
LAYERS = 10

def normalise(edge_index, size):
    adjacency = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.shape[1]),
        torch.Size((size, size))
    )
    degree = torch.mm(adjacency, torch.ones(size, 1)).squeeze()
    inv_degree = 1. / degree
    inv_degree_index = torch.index_select(inv_degree, 0, edge_index[1])
    matrix = torch.sparse_coo_tensor(
        edge_index,
        inv_degree_index,
        torch.Size((size, size))
    )
    return matrix

def load():
    print("loading node types...")
    nodes = torch.tensor(
        np.fromfile('tptp-graph/nodes.dat', dtype=np.uint8).astype(np.int64)
    )

    print("loading adjacency matrix...")
    from_index = np.concatenate((
        np.arange(nodes.shape[0], dtype=np.uint32),
        np.fromfile('tptp-graph/from.dat', dtype=np.uint32)
    ))
    to_index = np.concatenate((
        np.arange(nodes.shape[0], dtype=np.uint32),
        np.fromfile('tptp-graph/to.dat', dtype=np.uint32)
    ))
    edge_index_forward = torch.tensor(
        np.stack((to_index, from_index)).astype(np.int64)
    )
    edge_index_back = torch.tensor(
        np.stack((from_index, to_index)).astype(np.int64)
    )
    print("normalising matrix...")
    matrix_forward = normalise(edge_index_forward, nodes.shape[0])
    matrix_back = normalise(edge_index_back, nodes.shape[0])

    print("loading problem indices...")
    problem_index = torch.tensor(
        np.fromfile('tptp-graph/indices.dat', dtype=np.uint32)
            .astype(np.int64)
    )

    print("loading names...")
    problem_names = open('tptp-graph/problems.dat').read().split('\n')[:-1]

    print("OK")
    return nodes, matrix_forward, matrix_back, problem_index, problem_names

def train():
    model = Model(
        inputs=INPUT,
        dimensions=DIMENSIONS,
        channels=CHANNELS,
        layers=LAYERS
    )
    model.train()
    optimizer = Adam(model.parameters())
    nodes, matrix_forward, matrix_back, problem_index, problem_names = load()
    domains = [name[:3] for name in problem_names]
    metadata = [[name, domain] for name, domain in zip(problem_names, domains)]
    writer = SummaryWriter()

    step = 0
    while True:
        optimizer.zero_grad()
        mean, variance, y = model(
            nodes,
            matrix_forward,
            matrix_back,
            problem_index
        )

        reconstruction_loss = F.cross_entropy(y, nodes)
        divergence_loss = -0.5 * torch.mean(
            1 + torch.log(variance + EPS) - mean.pow(2) - variance
        )
        loss = reconstruction_loss + divergence_loss
        print("backwards pass...")
        loss.backward()
        print("...done")
        optimizer.step()

        writer.add_scalar(
            'loss/reconstruction',
            reconstruction_loss,
            step
        )
        writer.add_scalar(
            'loss/divergence',
            divergence_loss,
            step
        )
        writer.add_embedding(
            mean,
            metadata=metadata,
            metadata_header=['problem', 'domain'],
            global_step=step
        )

        torch.save(model.state_dict(), 'model.pt')
        step += 1

if __name__ == '__main__':
    train()
