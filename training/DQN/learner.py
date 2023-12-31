import abc
import copy
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from training.replay.ReplayBuffer import ReplayBuffer
from training.util.logger import logger
from training.util.torch_device import auto_get_device


@dataclass
class LearnerConfig:
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 1e-4
    optimizer_type: str = 'SGD'
    device: torch.device = auto_get_device()
    model_save_prefix: str = '/mnt/data3/rl-data/training_res'
    model_save_step: int = 1000
    # not start training until replay buffer has at least this number of transitions
    minimal_buffer_size: int = 1000


class Learner(abc.ABC):
    @abc.abstractmethod
    def step(self) -> Optional[float]:
        pass


class DQNLearner(Learner):
    def __init__(
            self,
            config: LearnerConfig,
            model: nn.Module,
            replay_buffer: ReplayBuffer,
            exp_name: str = None,
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.model = model.to(config.device)
        self.target_model = copy.deepcopy(model).to(config.device)
        self.optimizer = self.__get_optimizer()
        if exp_name is None:
            # generate name from current time and uuid
            exp_name = f'{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}-{uuid.uuid4()}'
        self.exp_name = exp_name
        if not os.path.exists(os.path.join(self.config.model_save_prefix, self.exp_name, 'models')):
            os.makedirs(os.path.join(self.config.model_save_prefix, self.exp_name, 'models'))
        self.step_cnt = 0

    def save_model(self, path, model, optimizer=None):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if os.path.exists(path):
            logger.warning(f'File {path} already exists, will overwrite it')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def load_model(self, path, model, optimizer=None):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer

    def __get_optimizer(self):
        if self.config.optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr)
        else:
            raise ValueError(f'Unknown optimizer type: {self.config.optimizer_type}')
        return optimizer

    def step(self) -> Optional[float]:
        """
        Run a single optimization step of a batch.
        """
        if len(self.replay_buffer.memory) < self.config.minimal_buffer_size:
            return None

        transitions = self.replay_buffer.sample(self.config.batch_size)
        state, model_output, reward, next_state, done = zip(*transitions)

        state_batch = torch.stack(state).to(self.config.device)
        action_batch = torch.stack(model_output).argmax(1).unsqueeze(1).to(self.config.device)
        reward_batch = torch.tensor(reward).to(self.config.device)
        next_state_batch = torch.stack(next_state).to(self.config.device)
        done_batch = torch.tensor(done).to(self.config.device)

        non_final_mask = done_batch != 0
        non_final_next_states = next_state_batch[non_final_mask]

        # Q(s_t, a)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # V(s_{t+1}) = max_a Q(s_{t+1}, a)
        next_state_values = torch.zeros(self.config.batch_size, device=self.config.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        # expected Q values
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch

        # Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # grad clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
        self.optimizer.step()

        self.step_cnt += 1

        if self.step_cnt % self.config.model_save_step == 0:
            path = os.path.join(self.config.model_save_prefix, self.exp_name, 'models', f'{self.step_cnt}.pt')
            logger.info(f'Save model (and optimizer) at step {self.step_cnt} into {path}')
            self.save_model(path, self.model, self.optimizer)

        return loss.item()

    def update_target_model(self):
        target_params = self.target_model.state_dict()
        params = self.model.state_dict()

        for k in target_params.keys():
            target_params[k] = (1 - self.config.tau) * target_params[k] + self.config.tau * params[k]

        self.target_model.load_state_dict(target_params)
