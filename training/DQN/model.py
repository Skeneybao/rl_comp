import abc
import random
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn

ActionType = Tuple[int, float, float]


class ModelOutputWrapper(abc.ABC):

    def __init__(self, model: nn.Module, refresh_model_steps: int = 32):
        self.model_base = model
        self.model = deepcopy(model).to('cpu')
        self.refresh_model_steps = refresh_model_steps
        self._refresh_count = 0

    def refresh_model(self):
        target_params = self.model_base.state_dict()
        self.model.load_state_dict(target_params)

    @staticmethod
    @abc.abstractmethod
    def get_output_shape():
        pass

    @abc.abstractmethod
    def select_action(self, observation, model_input: torch.tensor) -> Tuple[ActionType, torch.tensor, torch.tensor]:
        pass

    @abc.abstractmethod
    def random_action(self, observation, model_input) -> Tuple[ActionType, torch.tensor, torch.tensor]:
        pass


class Action11OutputWrapper(ModelOutputWrapper):
    buy_side = 0
    noop_side = 1
    sell_side = 2
    vol = 1.

    @staticmethod
    def get_output_shape():
        return 11

    def action_id_to_action(self, action_id, info) -> ActionType:
        # a4 -> a0 -> b0 -> b4 -> noop
        if action_id == 0:
            action = (self.buy_side, self.vol, info['ap4'])
        elif action_id == 1:
            action = (self.buy_side, self.vol, info['ap3'])
        elif action_id == 2:
            action = (self.buy_side, self.vol, info['ap2'])
        elif action_id == 3:
            action = (self.buy_side, self.vol, info['ap1'])
        elif action_id == 4:
            action = (self.buy_side, self.vol, info['ap0'])
        elif action_id == 5:
            action = (self.sell_side, self.vol, info['bp0'])
        elif action_id == 6:
            action = (self.sell_side, self.vol, info['bp1'])
        elif action_id == 7:
            action = (self.sell_side, self.vol, info['bp2'])
        elif action_id == 8:
            action = (self.sell_side, self.vol, info['bp3'])
        elif action_id == 9:
            action = (self.sell_side, self.vol, info['bp4'])
        elif action_id == 10:
            action = (self.noop_side, 0., 0.)
        else:
            raise ValueError('model output should between [0, 11)')
        return action

    def select_action(self, observation, model_input: torch.Tensor) -> Tuple[ActionType, torch.tensor, torch.tensor]:
        # 0. inference
        with torch.no_grad():
            model_output = self.model(model_input)
        # 1. postprocess output
        action = self.action_id_to_action(model_output.argmax(-1).item(), observation)

        self._refresh_count += 1
        if self._refresh_count % self.refresh_model_steps == 0:
            self.refresh_model()

        return action, model_input, model_output

    def random_action(self, observation, model_input) -> Tuple[ActionType, torch.tensor, torch.tensor]:
        action_id = random.randrange(0, 11)
        action = self.action_id_to_action(action_id, observation)
        model_output = torch.zeros(11, dtype=torch.float)
        model_output[action_id] = 1
        return action, model_input, model_output
