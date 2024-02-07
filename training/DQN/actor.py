import random
import os
import csv
from dataclasses import dataclass
from typing import Callable

import math
import torch
from torch import nn
import numpy as np

from training.model_io.output_wrapper import ModelOutputWrapper
from training.model_io.featureEngine import FeatureEngine
from training.env.trainingEnv import TrainingStockEnv
from training.replay.ReplayBuffer import ReplayBuffer
from training.util.report_running_time import report_time
from training.util.validate_action import validate_action
from training.util.explicit_control import ExplicitControlConf
from training.util.logger import log_states


@dataclass
class ActorConfig:
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 1000


def cal_epsilon(
        config: ActorConfig,
        steps_done: int,
):
    eps_threshold = config.eps_end + (config.eps_start - config.eps_end) * math.exp(-1. * steps_done / config.eps_decay)
    return eps_threshold


def if_epsilon_greedy(
        config: ActorConfig,
        steps_done: int,
):
    """
    Decide whether to use epsilon greedy strategy. If return True, use epsilon greedy strategy.
    """
    sample = random.random()
    eps_threshold = cal_epsilon(config, steps_done)
    if sample < eps_threshold:
        return True
    else:
        return False


class Actor:
    def __init__(
            self,
            env: TrainingStockEnv,
            feature_engine: FeatureEngine,
            output_wrapper: ModelOutputWrapper,
            replay_buffer: ReplayBuffer,
            config: ActorConfig,
            explicit_config: ExplicitControlConf = ExplicitControlConf(-float('inf')),
    ):
        self.env = env
        self.feature_engine = feature_engine
        self.this_obs, self.last_reward, _ = self.env.reset()
        self.this_state = self.feature_engine.get_feature(self.this_obs) 
        self.output_wrapper = output_wrapper
        self.replay_buffer = replay_buffer
        self.config = config
        self.explicit_config = explicit_config

    @report_time(100000)
    def step(self):
        warming_up = self.this_obs['warming-up']
        if not self.this_obs['warming-up']:
            if if_epsilon_greedy(self.config, self.env.step_cnt):
                action, _, model_output = self.output_wrapper.random_action(self.this_obs, self.this_state)
            else:
                action, _, model_output = self.output_wrapper.select_action(self.this_obs, self.this_state)

            valid_action, is_invalid = validate_action(self.this_obs, action, max_position=self.feature_engine.max_position, signal_risk_thresh=self.explicit_config.signal_risk_thresh)
            next_obs, reward, done = self.env.step(valid_action)
            next_state = self.feature_engine.get_feature(next_obs)
            if not self.this_obs['eventTime'] > 144700000:
                self.replay_buffer.push([self.this_state, model_output, reward, next_state, done])
        else:
            action = (1, 0, 0)
            valid_action = (1, 0 ,0)
            model_output = torch.zeros(self.output_wrapper.get_output_shape(), dtype=torch.float)
            next_obs, reward, done = self.env.step(valid_action)
            next_state = self.feature_engine.get_feature(next_obs)
        log_states(self.env, self.this_obs, self.feature_engine, self.this_state, reward, 1 - action[0], 1 - valid_action[0], model_output.numpy().flatten())

        self.last_reward = reward
        self.this_obs = next_obs
        self.this_state = next_state
        return not warming_up
