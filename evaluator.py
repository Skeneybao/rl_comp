import os
from dataclasses import dataclass
from typing import Callable, Dict

import torch

from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNN
from training.model_io.featureEngine import FeatureEngineVersion1
from training.model_io.output_wrapper import Action11OutputWrapper
from training.model_io.output_wrapper import ActionType
from training.reward.normalized_net_return import cal_reward
from training.util.validate_action import validate_action


@dataclass
class EvaluatorConfig:
    training_res_path: str
    model_name: str
    data_path: str = '/home/rl-comp/Git/rl_comp/env/stock_raw/data'
    date: str = 'ALL'
    reward_fn: Callable[[int, Dict, Dict, ActionType], float] = cal_reward


def evaluate_model(config: EvaluatorConfig):
    eval_res_path = os.path.join(config.training_res_path, f"eval_result_{config.model_name}")
    if not os.path.exists(eval_res_path):
        os.makedirs(eval_res_path)

    env = TrainingStockEnv(
        mode='ordered',
        data_path=config.data_path,
        dates=config.date,
        save_metric_path=eval_res_path,
        save_daily_metric=True,
        save_code_metric=True,

    )

    feature_engine = FeatureEngineVersion1()
    model = DNN(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape())
    checkpoint = torch.load(os.path.join(config.training_res_path, 'models', config.model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    model_output_wrapper = Action11OutputWrapper(model)

    obs, _, _ = env.reset()

    while env.reset_cnt <= len(env):
        state = feature_engine.get_feature(obs)
        action, _, _ = model_output_wrapper.select_action(obs, state)
        valid_action, is_invalid = validate_action(obs, action)
        obs, _, _ = env.step(valid_action)


if __name__ == '__main__':
    config = EvaluatorConfig(
        data_path='/home/rl-comp/Git/rl_comp/env/stock_raw/data',
        date='ALL',
        training_res_path='/mnt/data3/rl-data/training_res/20240104:092411-b3c43eac',
        model_name='120000.pt',
    )

    evaluate_model(config)
