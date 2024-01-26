import pandas as pd
import numpy as np
import torch
from torch import nn
from training.model_io.featureEngine import *
from training.model_io.output_wrapper import *
from training.model.DNN import *
from training.util.validate_action import validate_action


feature_engine = FeatureEngineVersion1()
model = DNN(
    input_dim=feature_engine.get_input_shape(),
    output_dim=Action11OutputWrapper.get_output_shape(),
    hidden_dim=[64, 64])
checkpoint = torch.load('/mnt/data3/rl-data/training_res/Champions/1/models/6860000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model_output_wrapper = Action11OutputWrapper(model)


def my_controller(observation, action_space, is_act_continuous=False):
    obs = observation['observation']
    state = feature_engine.get_feature(obs)
    action, _, _ = model_output_wrapper.select_action(obs, state)
    (sd, vol, price), is_invalid = validate_action(obs, action)
    
    if sd == 0:
        side = [1, 0, 0]
    elif sd == 1:
        side = [0, 1, 0]
    else:
        side = [0, 0, 1]
    return [side, [vol], [price]]

