import sys
from pathlib import Path

import torch

from data import Problems
from model import Model

if __name__ == "__main__":
    model = Model().to("cuda")
    model.eval()
    model.load_state_dict(torch.load("model.pt"))

    print("problem\tdomain", file=sys.stderr)
    for problem, nodes, sources, targets in Problems():
        with torch.no_grad():
            name = Path(problem).stem
            print(name + "\t" + name[:3], file=sys.stderr)
            embedding = model.encode(
                nodes.to("cuda"), sources.to("cuda"), targets.to("cuda")
            )
            print("\t".join(str(float(x)) for x in embedding))
