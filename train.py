import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import Problems
from model import Model

LR = 5e-3
GAMMA = 0.9995
MOMENTUM = 0.9
BATCH = 32
DIVERGENCE_WEIGHT = 0.1

if __name__ == '__main__':
    model = Model().to('cuda')
    optimiser = SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        nesterov=True
    )
    scheduler = ExponentialLR(optimiser, gamma=GAMMA)
    writer = SummaryWriter()
    step = 0
    problems = Problems()
    while True:
        loader = DataLoader(
            problems,
            collate_fn=lambda x: x[0],
            num_workers=1,
            pin_memory=True
        )
        for problem, nodes, sources, targets in tqdm(loader):
            nodes = nodes.to('cuda')
            sources = sources.to('cuda')
            targets = targets.to('cuda')
            mean, logvar, reconstruction = model(nodes, sources, targets)

            reconstruction_loss = F.cross_entropy(reconstruction, nodes)
            divergence_loss = -0.5 * torch.mean(
                1 + logvar - mean.pow(2) - logvar.exp()
            )
            loss = reconstruction_loss + DIVERGENCE_WEIGHT * divergence_loss
            (loss / BATCH).backward()

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
            writer.add_scalar(
                'loss/loss',
                loss,
                step
            )

            step += 1
            if step % BATCH == 0:
                writer.add_scalar(
                    'training/LR',
                    optimiser.param_groups[0]['lr'],
                    step
                )
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()

        torch.save(model.state_dict(), 'save.pt')
