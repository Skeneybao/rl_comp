import pandas as pd
import numpy as np
import torch
from torch import nn

from training.model_io.env_info_appender import EnvInfoAppender
from training.model_io.featureEngine import *
from training.model_io.output_wrapper import *
from training.model.DNN import *
from training.util.validate_action import validate_action
from training.util.explicit_control import ExplicitControlConf

max_position = 300

feature_engine = FeatureEngineVersion4(max_position=max_position)
model = DNN(
    input_dim=feature_engine.get_input_shape(),
    output_dim=Action3OutputWrapper.get_output_shape(),
    hidden_dim=[16, 16, 16])
checkpoint = torch.load('/mnt/data3/rl-data/training_res/fkh3gv81/Y7Ysb/models/3000000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model_output_wrapper = Action3OutputWrapper(model)
explicit_config = ExplicitControlConf(signal_risk_thresh=0)
env_info_appender = EnvInfoAppender(max_position=max_position)


def my_controller(observation, action_space, is_act_continuous=False):
    obs = observation['observation']
    env_info_appender.accumulate((obs['ap0'] + obs['bp0']) / 2, obs['signal0'], obs['signal1'], obs['signal2'], 0)
    obs = {**obs, **env_info_appender.get_info(obs)}
    if observation['new_game']:
        env_info_appender.reset()
    state = feature_engine.get_feature(obs)
    action, _, _ = model_output_wrapper.select_action(obs, state)
    sd, vol, price = action
    if sd == 0:
        vol = obs['av0']
    elif sd == 2:
        vol = obs['bv0']
    (sd, vol, price), is_invalid = validate_action(obs, (sd, vol, price), max_position=feature_engine.max_position,
                                                   signal_risk_thresh=explicit_config.signal_risk_thresh)
    if sd == 0:
        side = [1, 0, 0]
    elif sd == 1:
        side = [0, 1, 0]
    else:
        side = [0, 0, 1]
    return [side, [vol], [price]]

