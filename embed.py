import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from data import Problems
from model import Model

if __name__ == "__main__":
    model = Model().to("cuda")
    model.eval()
    model.load_state_dict(torch.load("model.pt"))
    embeddings = dict()
    for problem, nodes, sources, targets in tqdm(Problems()):
        with torch.no_grad():
            name = Path(problem).stem
            embedding = model.encode(
                nodes.to("cuda"), sources.to("cuda"), targets.to("cuda")
            )
            embeddings[name] = embedding
    with open("embeddings.pkl", "wb") as pickle_file:
        pickle.dump(embeddings, pickle_file)
