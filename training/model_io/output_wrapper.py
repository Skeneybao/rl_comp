import abc
import random
from copy import deepcopy
from typing import Tuple, Dict, Type

import torch
import torch.nn as nn

# side, vol, price
# side = 0: buy
# side = 1: noop
# side = 2: sell
ActionType = Tuple[int, float, float]


class ModelOutputWrapper(abc.ABC):

    def __init__(self, model: nn.Module, refresh_model_steps: int = 32, device: str = 'cpu'):
        self.model_base = model
        self.device = device
        self.model = deepcopy(model).to(device)
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

    def action_id_to_action(self, action_id: int, obs: Dict) -> ActionType:
        # a4 -> a0 -> b0 -> b4 -> noop
        if 0 <= action_id < 5:
            vol = 5 - self.vol * action_id
            price = obs['ap4']
            action = (self.buy_side, vol, price)

        elif 5 <= action_id < 10:
            vol = self.vol * (action_id - 4)
            price = obs['bp4']
            action = (self.sell_side, vol, price)
        elif action_id == 10:
            action = (self.noop_side, 0., 0.)
        else:
            raise ValueError(f'model output should between [0, {self.get_output_shape()})')
        return action

    def select_action(self, observation, model_input: torch.Tensor) -> Tuple[ActionType, torch.tensor, torch.tensor]:

        if observation['eventTime'] > 145500000:
            return (self.noop_side, 0., 0.), None, None

        # 0. inference
        with torch.no_grad():
            model_output = self.model(model_input.to(self.device))
        # 1. postprocess output
        action = self.action_id_to_action(model_output.argmax(-1).item(), observation)

        self._refresh_count += 1
        if self._refresh_count % self.refresh_model_steps == 0:
            self.refresh_model()

        return action, model_input, model_output

    def random_action(self, observation, model_input) -> Tuple[ActionType, torch.tensor, torch.tensor]:

        if observation['eventTime'] > 145500000:
            return (self.noop_side, 0., 0.), None, None
        action_id = random.randrange(0, 11)
        action = self.action_id_to_action(action_id, observation)
        model_output = torch.zeros(11, dtype=torch.float)
        model_output[action_id] = 1
        return action, model_input, model_output


class Action3OutputWrapper(ModelOutputWrapper):
    buy_side = 0
    noop_side = 1
    sell_side = 2
    vol = 1.

    @staticmethod
    def get_output_shape():
        return 3

    def action_id_to_action(self, action_id: int, obs: Dict) -> ActionType:
        # a4 -> a0 -> b0 -> b4 -> noop
        if action_id == 0:
            vol = self.vol
            price = obs['ap4']
            action = (self.buy_side, vol, price)

        elif 5 <= action_id < 10:
            vol = - self.vol
            price = obs['bp4']
            action = (self.sell_side, vol, price)
        elif action_id == 10:
            action = (self.noop_side, 0., 0.)
        else:
            raise ValueError(f'model output should between [0, {self.get_output_shape()})')
        return action

    def select_action(self, observation, model_input: torch.Tensor) -> Tuple[ActionType, torch.tensor, torch.tensor]:

        if observation['eventTime'] > 145500000:
            return (self.noop_side, 0., 0.), None, None

        # 0. inference
        with torch.no_grad():
            model_output = self.model(model_input.to(self.device))
        # 1. postprocess output
        action = self.action_id_to_action(model_output.argmax(-1).item(), observation)

        self._refresh_count += 1
        if self._refresh_count % self.refresh_model_steps == 0:
            self.refresh_model()

        return action, model_input, model_output

    def random_action(self, observation, model_input) -> Tuple[ActionType, torch.tensor, torch.tensor]:

        if observation['eventTime'] > 145500000:
            return (self.noop_side, 0., 0.), None, None
        action_id = random.randrange(0, 3)
        action = self.action_id_to_action(action_id, observation)
        model_output = torch.zeros(3, dtype=torch.float)
        model_output[action_id] = 1
        return action, model_input, model_output


def get_output_wrapper(name: str) -> Type[ModelOutputWrapper]:
    if name == 'action11':
        return Action11OutputWrapper
    elif name == 'action3':
        return Action3OutputWrapper
    else:
        raise ValueError(f'unknown output wrapper {name}')
