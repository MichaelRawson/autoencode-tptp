import torch
from pathlib import Path
import sys

from model import Model
from data import problems

if __name__ == '__main__':
    model = Model()
    model.eval()
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))

    for problem, nodes, sources, targets in problems(set(), randomise=False):
        with torch.no_grad():
            name = Path(problem).stem 
            print(name + "\t" + name[:3], file=sys.stderr)
            embedding = model.encode(nodes, sources, targets)
            print("\t".join(str(float(x)) for x in embedding))
