from tptp_graph import graph_of
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

from model import Model
from data import problems

if __name__ == '__main__':
    model = Model().to('cuda')
    model.eval()
    model.load_state_dict(torch.load('model.pt'))

    nodes, sources, targets = graph_of(sys.argv[1])
    nodes = torch.tensor(nodes)
    sources = torch.tensor(sources)
    targets = torch.tensor(targets)
    with torch.no_grad():
        _, _, logits = model(
            nodes.to('cuda'),
            sources.to('cuda'),
            targets.to('cuda')
        )
        reconstruction = torch.softmax(5 * logits, dim=0).to('cpu')
        original = torch.eye(reconstruction.shape[1])[nodes]
        #comparison = torch.cat((original, reconstruction), dim=1)
        plt.imshow(reconstruction)
        plt.show()
