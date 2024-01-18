import random
import os
import csv
from dataclasses import dataclass
from typing import Callable

import math
import torch
from torch import nn

from training.model_io.output_wrapper import ModelOutputWrapper
from training.model_io.featureEngine import FeatureEngine
from training.env.trainingEnv import TrainingStockEnv
from training.replay.ReplayBuffer import ReplayBuffer
from training.util.validate_action import validate_action


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
    ):
        self.env = env
        self.feature_engine = feature_engine
        self.this_obs, _, _ = self.env.reset()
        self.this_state = self.feature_engine.get_feature(self.this_obs) 
        self.output_wrapper = output_wrapper
        self.replay_buffer = replay_buffer
        self.config = config

    def step(self):

        self.log_states()

        if if_epsilon_greedy(self.config, self.env.step_cnt):
            action, _, model_output = self.output_wrapper.random_action(self.this_obs, self.this_state)
        else:
            action, _, model_output = self.output_wrapper.select_action(self.this_obs, self.this_state)

        valid_action, is_invalid = validate_action(self.this_obs, action)
        next_obs, reward, done = self.env.step(valid_action)
        next_state = self.feature_engine.get_feature(next_obs)
        if not self.this_obs['eventTime'] > 145500000:
            self.replay_buffer.push([self.this_state, model_output, reward, next_state, done])
        
        self.this_obs = next_obs
        self.this_state = next_state

    def log_states(self):
        current_code = self.this_obs['code']
        if self.env.save_code_metric and current_code in self.env.codes_to_log:
            output_file_path = os.path.join(self.env.save_metric_path, 'code_metric',f"{self.env.date}_{current_code}_states.csv")
            file_exists = os.path.exists(output_file_path)
            with open(output_file_path, 'a', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=self.feature_engine.feature_names)
                if not file_exists:
                    csv_writer.writeheader()
                csv_writer.writerow(dict(zip(self.feature_engine.feature_names, self.this_state.numpy())))