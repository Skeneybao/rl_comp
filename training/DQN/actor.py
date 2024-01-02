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
            replay_buffer: ReplayBuffer,
            config: ActorConfig,
    ):
        self.env = new_env_fn()
        self.this_obs, _, _ = self.env.reset()
        self.feature_engine = feature_engine
        self.output_wrapper = output_wrapper
        self.replay_buffer = replay_buffer
        self.config = config

    def step(self):
        obs = self.this_obs
        state = self.feature_engine.get_feature(obs)
        if if_epsilon_greedy(self.config, self.env.step_cnt):
            action, state, model_output = self.output_wrapper.random_action(obs, state)
        else:
            action, state, model_output = self.output_wrapper.select_action(obs, state)

        valid_action, is_invalid = validate_action(obs, action)
        next_obs, reward, done = self.env.step(valid_action)
        next_state = self.feature_engine.get_feature(next_obs)
        self.replay_buffer.push([state, model_output, reward, next_state, done])
        self.this_obs = next_obs
