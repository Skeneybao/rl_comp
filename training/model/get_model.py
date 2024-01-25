from typing import Type

from training.model.DNN import *


def get_model(name: str) -> Type[nn.Module]:
    if name == 'dnn':
        return DNN
    elif name == 'dnn_11_output':
        return DNN_11_output
    else:
        raise ValueError(f'Unknown model name: {name}')
