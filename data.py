from glob import glob
from pathlib import Path
from random import shuffle

import torch
from torch.utils.data import IterableDataset
from tptp_graph import graph_of


class Problems(IterableDataset):
    def __init__(self, paths = glob('Problems/**/*+*.p'), randomise=True):
        self.blacklist = set()
        self.paths = paths
        self.randomise = randomise

    def __iter__(self):
        if self.randomise:
            shuffle(self.paths)
        for path in self.paths:
            if path in self.blacklist:
                continue
            nodes, sources, targets = graph_of(path)
            if nodes.shape[0] > 50000:
                self.blacklist.add(path)
                continue
            nodes = torch.tensor(nodes)
            sources = torch.tensor(sources)
            targets = torch.tensor(targets)
            path = Path(path)
            problem = path.stem
            yield problem, nodes, sources, targets

    def __len__(self):
        return len(self.paths)
