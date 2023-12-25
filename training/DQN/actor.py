import random
from dataclasses import dataclass
from typing import Callable

import math
import torch
from torch import nn

from training.DQN.model import ModelOutputWrapper
from training.env.featureEngine import FeatureEngine
from training.env.trainingEnv import TrainingStockEnv
from training.replay.ReplayBuffer import ReplayBuffer
from training.util.validate_action import validate_action


@dataclass
class ActorConfig:
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 1000


def if_epsilon_greedy(
        config: ActorConfig,
        steps_done: int,
):
    """
    Decide whether to use epsilon greedy strategy. If return True, use epsilon greedy strategy.
    """
    sample = random.random()
    eps_threshold = config.eps_end + (config.eps_start - config.eps_end) * math.exp(-1. * steps_done / config.eps_decay)
    if sample < eps_threshold:
        return True
    else:
        return False


class Actor:
    def __init__(
            self,
            new_env_fn: Callable[[], TrainingStockEnv],
            feature_engine: FeatureEngine,
            output_wrapper: ModelOutputWrapper,
            model: nn.Module,
            replay_buffer: ReplayBuffer,
            config: ActorConfig,
    ):
        self.env = new_env_fn()
        self.this_state, _, _ = self.env.reset()
        self.feature_engine = feature_engine
        self.model = model
        self.output_wrapper = output_wrapper
        self.replay_buffer = replay_buffer
        self.config = config

    def step(self):
        state = self.this_state
        model_input = self.feature_engine.get_feature(state)
        if if_epsilon_greedy(self.config, self.env.step_cnt):
            action, model_input, model_output = self.output_wrapper.random_action(state, model_input)
        else:
            action, model_input, model_output = self.output_wrapper.select_action(
                self.model, state, model_input)

        valid_action, is_invalid = validate_action(state, action)
        next_state, reward, done = self.env.step(valid_action)
        next_model_input = self.feature_engine.get_feature(next_state)
        self.replay_buffer.push([state, action, reward, next_state, done, model_input, model_output, next_model_input])
        self.this_state = next_state
