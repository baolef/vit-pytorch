# Created by Baole Fang at 4/2/24

import vit_pytorch
from torch.nn.parallel import DataParallel


def make_model(config, gpus, state):
    model = getattr(vit_pytorch, config['type'])(**config['args'])
    if len(gpus) > 1:
        model = DataParallel(model, device_ids=gpus)
    if state:
        if len(gpus) > 1:
            model.module.load_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
    return model.to(gpus[0])

