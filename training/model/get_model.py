from typing import Type

from training.model.DNN import *


def get_model(name: str) -> Type[nn.Module]:
    if name == 'dnn':
        return DNN
    else:
        raise ValueError(f'Unknown model name: {name}')
