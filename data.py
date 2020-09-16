import torch
from torch.utils.data import IterableDataset
from tptp_graph import graph_of
from glob import glob
from pathlib import Path
from random import shuffle

def problems(blacklist, randomise=True):
    paths = glob('Problems/**/*+*.p')
    if randomise:
        shuffle(paths)
    for path in paths:
        if path in blacklist:
            continue
        nodes, sources, targets = graph_of(path)
        if nodes.shape[0] > 50000:
            blacklist.add(path)
            continue
        nodes = torch.tensor(nodes)
        sources = torch.tensor(sources)
        targets = torch.tensor(targets)
        path = Path(path)
        problem = path.stem
        yield problem, nodes, sources, targets

class Problems(IterableDataset):
    def __init__(self):
        self.blacklist = set()
        super().__init__()

    def __iter__(self):
        return problems(self.blacklist)
