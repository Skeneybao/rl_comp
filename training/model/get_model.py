from typing import Type

from training.model.Attn import Attn
from training.model.DNN import *


def get_model(name: str) -> Type[nn.Module]:
    if name == 'dnn':
        return DNN
    elif name == 'dnn_11_output':
        return DNN_11_output
    elif name == 'attn':
        return Attn
    elif name == 'full_pos_dnn':
        return FullPosDNN
    else:
        raise ValueError(f'Unknown model name: {name}')
