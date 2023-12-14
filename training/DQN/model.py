import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

ActionType = Tuple[list, list, list]


@dataclass
class ActorConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int


class Actor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModelIOWrapper:
    buy_side = [1, 0, 0]
    sell_side = [0, 0, 1]
    noop_side = [0, 1, 0]
    vol = 1

    def __init__(self, model):
        self.model = model

    def action_id_to_action(self, action_id, info):
        # a4 -> a0 -> b0 -> b4 -> noop
        if action_id == 0:
            action = self.buy_side, [self.vol], [info['ap4']]
        elif action_id == 1:
            action = self.buy_side, [self.vol], [info['ap3']]
        elif action_id == 2:
            action = self.buy_side, [self.vol], [info['ap2']]
        elif action_id == 3:
            action = self.buy_side, [self.vol], [info['ap1']]
        elif action_id == 4:
            action = self.buy_side, [self.vol], [info['ap0']],
        elif action_id == 5:
            action = self.sell_side, [self.vol], [info['bp0']]
        elif action_id == 6:
            action = self.sell_side, [self.vol], [info['bp1']]
        elif action_id == 7:
            action = self.sell_side, [self.vol], [info['bp2']]
        elif action_id == 8:
            action = self.sell_side, [self.vol], [info['bp3']]
        elif action_id == 9:
            action = self.sell_side, [self.vol], [info['bp4']]
        elif action_id == 10:
            action = self.noop_side, [0], [0]
        else:
            raise ValueError('model output should between [0, 11)')
        return action

    def wrap_inference_single(self, state) -> Tuple[ActionType, torch.tensor, torch.tensor]:
        # public info
        info = state[-1]['observation']
        # 0. preprocess into
        model_input = torch.tensor(list(info.values()))
        # 1. inference
        with torch.no_grad():
            model_output = self.model(model_input)
        # 2. postprocess output
        action = self.action_id_to_action(model_output.argmax(-1).item(), info)

        return action, model_input, model_output

    def random_action(self, state):
        info = state[-1]['observation']
        action_id = random.randrange(0, 11)
        action = self.action_id_to_action(action_id, info)
        return action
