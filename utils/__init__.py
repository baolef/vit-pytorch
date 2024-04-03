# Created by Baole Fang at 4/2/24

from .dataset import make_dataloaders
from .model import make_model
from .optimizer import make_optimizer
from .scheduler import make_scheduler
import torch

__all__ = ['prepare']

def prepare(config, gpus):
    train_loader, test_loader = make_dataloaders(config['dataset'])
    if config['resume']:
        state = torch.load(config['resume'])
    else:
        state = None
    model = make_model(config['model'], gpus, state)
    optimizer = make_optimizer(config['optimizer'], model, state)
    scheduler = make_scheduler(config['scheduler'], optimizer, state)

    return train_loader, test_loader, model, optimizer, scheduler
