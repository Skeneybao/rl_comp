import os
import csv
from dataclasses import dataclass
from typing import Callable, Dict, Type, Any

import numpy as np
import torch
from torch import nn

from training.model_io.featureEngine import FeatureEngineVersion3_Simple, FeatureEngine
from training.model.DNN import DNN
from training.env.trainingEnv import TrainingStockEnv
from training.model_io.output_wrapper import Action11OutputWrapper, Action3OutputWrapper, ModelOutputWrapper
from training.model_io.output_wrapper import ActionType
from training.reward.rewards import *
from training.util.logger import logger
from training.util.validate_action import validate_action


@dataclass(frozen=True)
class EvaluatorConfig:
    training_res_path: str
    model_name: str
    feature_engine_type: Type[FeatureEngine]
    feature_engine_param: Dict[str, Any]
    model_type: Type[nn.Module]
    model_param: Dict[str, Any]
    output_wrapper_type: Type[ModelOutputWrapper]
    reward_fn: Callable[[int, Dict, Dict, ActionType], float]
    data_path: str = '/home/rl-comp/Git/rl_comp/env/stock_raw/data'
    date: str = 'ALL'

    device: str = 'cpu'


def evaluate_model(config: EvaluatorConfig):
    eval_res_path = os.path.join(config.training_res_path, f"eval_result_{config.model_name}")
    if not os.path.exists(eval_res_path):
        os.makedirs(eval_res_path)

    logger.info(f'evaluating model {config.model_name} on {config.date}')
    logger.info(f'full config: {config}')
    feature_engine = config.feature_engine_type(**config.feature_engine_param)

    env = TrainingStockEnv(
        mode='ordered',
        data_path=config.data_path,
        dates=config.date,
        save_metric_path=eval_res_path,
        save_daily_metric=True,
        save_code_metric=True,
        reward_fn=config.reward_fn,
        max_postion=feature_engine.max_position,    
        )

    model = config.model_type(
        input_dim=feature_engine.get_input_shape(),
        output_dim=config.output_wrapper_type.get_output_shape(),
        **config.model_param)
    checkpoint = torch.load(os.path.join(config.training_res_path, 'models', config.model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    model_output_wrapper = config.output_wrapper_type(model, device=config.device)

    obs, reward, _ = env.reset()

    while env.reset_cnt <= len(env):
        state = feature_engine.get_feature(obs)
        log_states(env, obs, feature_engine, state, reward)
        if obs['warming-up']:
            valid_action = (1, 0, 0)
        else:
            action, _, _ = model_output_wrapper.select_action(obs, state)
            valid_action, is_invalid = validate_action(obs, action, max_position=feature_engine.max_position)
        obs, reward, _ = env.step(valid_action)

    logger.info(f'evaluating model {config.model_name} on {config.date} done.')

    return env.compute_final_stats()


def log_states(env, obs, feature_engine, state, reward):

    current_code = obs['code']
    if env.save_code_metric and current_code in env.codes_to_log:
        output_file_path = os.path.join(env.save_metric_path, 'code_metric',f"{env.date}_{current_code}_states.csv")
        file_exists = os.path.exists(output_file_path)
        with open(output_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=feature_engine.feature_names + ['reward'])
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow(dict(zip(feature_engine.feature_names + ['reward'], np.append(state.numpy(), reward))))
        
if __name__ == '__main__':
    
    config = EvaluatorConfig(
        data_path='/home/rl-comp/Git/rl_comp/env/stock_raw/data',
        date='ALL',
        training_res_path='/mnt/data3/rl-data/training_res/7u6e3hgm/HkUyH/',
        model_name='5080000.pt',
        feature_engine_type=FeatureEngineVersion3_Simple,
        feature_engine_param={'max_position': 10},
        model_type=DNN,
        model_param={'hidden_dim': [32, 32]},
        output_wrapper_type=Action3OutputWrapper,
        reward_fn=normalized_net_return,
    )

    res = evaluate_model(config)
    print(res)
