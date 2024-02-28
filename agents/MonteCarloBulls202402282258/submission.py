#####################
# env_info_wrapper
#####################
import abc
import os
from dataclasses import field, dataclass
from typing import Optional

import math
import numpy as np


class Accumulator(abc.ABC):
    @abc.abstractmethod
    def accumulate(self, data):
        pass

    @abc.abstractmethod
    def get_data(self):
        pass


class RollingSumAccumulator(Accumulator):
    rolling_window = 240

    def __init__(self):
        self.partial_sum = 0
        self.count = 0
        self.items = []

    def accumulate(self, data):
        self.partial_sum += data
        self.count += 1
        self.items.append(data)
        if self.count > self.rolling_window:
            self.partial_sum -= self.items.pop(0)
            self.count -= 1

    def get_data(self):
        return self.partial_sum / self.count


class RollingStdAccumulator(Accumulator):
    rolling_window = 240

    def __init__(self):
        self.partial_sum = 0.
        self.partial_sum_sq = 0.
        self.count = 0
        self.items = []

    def accumulate(self, data):
        self.partial_sum += data
        self.partial_sum_sq += data ** 2
        self.count += 1
        self.items.append(data)
        if self.count > self.rolling_window:
            item_to_pop = self.items.pop(0)
            self.partial_sum -= item_to_pop
            self.partial_sum_sq -= item_to_pop ** 2
            self.count -= 1

    def get_data(self):
        if self.count < 2:
            return 1

        diff = self.partial_sum_sq - self.partial_sum ** 2 / self.count
        if -1e-2 < diff < 0:
            return 0
        return math.sqrt(diff / (self.count - 1))


@dataclass
class InfoAccumulator:
    sig0_queue: list = field(default_factory=list)
    sig1_queue: list = field(default_factory=list)
    sig2_queue: list = field(default_factory=list)
    sig0_sum: RollingSumAccumulator = field(default_factory=RollingSumAccumulator)
    sig1_sum: RollingSumAccumulator = field(default_factory=RollingSumAccumulator)
    sig2_sum: RollingSumAccumulator = field(default_factory=RollingSumAccumulator)
    mid_price_std: RollingStdAccumulator = field(default_factory=RollingStdAccumulator)
    code_reward_accum: float = 0
    daily_reward_accum: float = 0

    def log(self, mid_price, sig0, sig1, sig2, reward):
        self.sig0_queue.append(sig0)
        self.sig1_queue.append(sig1)
        self.sig2_queue.append(sig2)
        self.sig0_sum.accumulate(sig0)
        self.sig1_sum.accumulate(sig1)
        self.sig2_sum.accumulate(sig2)
        self.mid_price_std.accumulate(mid_price)
        self.code_reward_accum += reward
        self.daily_reward_accum += reward

    def get_data(self, obs):
        info = {}
        if len(self.sig0_queue) > 240:
            info['signal0_rank'] = get_rank(self.sig0_queue[-240:], obs['signal0']) / 240
            info['signal1_rank'] = get_rank(self.sig1_queue[-240:], obs['signal1']) / 240
            info['signal2_rank'] = get_rank(self.sig2_queue[-240:], obs['signal2']) / 240
            info['signal0_mean'] = self.sig0_sum.get_data()
            info['signal1_mean'] = self.sig1_sum.get_data()
            info['signal2_mean'] = self.sig2_sum.get_data()
            info['mid_price_std'] = self.mid_price_std.get_data() / obs['ap0_t0']
            info['warming-up'] = False
        else:
            info['signal0_rank'] = 0.5
            info['signal1_rank'] = 0.5
            info['signal2_rank'] = 0.5
            info['signal0_mean'] = 0.
            info['signal1_mean'] = 0.
            info['signal2_mean'] = 0.
            info['mid_price_std'] = 1.
            info['warming-up'] = True
        return info


class EnvInfoAppender:
    def __init__(self, max_position):
        self.info_acc = InfoAccumulator()
        self.max_position = max_position

    def accumulate(self, mid_price, sig0, sig1, sig2, reward):
        self.info_acc.log(mid_price, sig0, sig1, sig2, reward)

    def get_info(self, obs):
        info = self.info_acc.get_data(obs)

        if obs['code_net_position'] == self.max_position:
            info['full_pos'] = 1
        elif obs['code_net_position'] == -self.max_position:
            info['full_pos'] = -1
        else:
            info['full_pos'] = 0

        return info

    def reset(self, info_acc: Optional[InfoAccumulator] = None):
        if info_acc is None:
            self.info_acc = InfoAccumulator()
        else:
            self.info_acc = info_acc

#####################
# DNN
#####################
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(),
    'gelu': nn.GELU(approximate='tanh'),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
}


@dataclass
class DNNModelConfig:
    input_dim: int
    hidden_dim: List[int]
    output_dim: int
    activation: str = 'gelu'
    output_activation: Optional[str] = None


class DNN(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: List[int],
            output_dim: int,
            activation: str = 'gelu',
            output_activation: Optional[str] = None
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        self.fc_out = nn.Linear(hidden_dim[-1], output_dim)

        self.activation = activations[activation]
        if output_activation is not None:
            self.output_activation = activations[output_activation]
        else:
            self.output_activation = None

    def forward(self, x):
        x = self.activation(self.fc_in(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        x = self.fc_out(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


#####################
# feature engine
#####################
import abc
from typing import List

import numpy as np
import torch


class FeatureEngine(abc.ABC):
    @abc.abstractmethod
    def get_input_shape(self):
        pass

    @abc.abstractmethod
    def get_feature(self, observation) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def feature_names(self) -> List[str]:
        pass


class FeatureEngineExample(FeatureEngine):

    @property
    def feature_names(self) -> List[str]:
        return ['feature1', 'feature2', 'feature3']

    def __init__(self, feature_to_use=None):
        pass

    def get_input_shape(self):
        return 3

    def get_feature(self, observation):
        feature_array = torch.tensor([
            self.feature1(observation),
            self.feature2(observation),
            self.feature3(observation),
        ])

    def feature1(self, observation):
        return 1

    def feature2(self, observation):
        return 2

    def feature3(self, observation):
        return 3


class FeatureEngineVersion1(FeatureEngine):

    def __init__(self, max_position=300):
        self.max_position = max_position

    def get_input_shape(self):
        return 17

    def get_feature(self, observation) -> torch.Tensor:
        mid_price = (observation['ap0'] + observation['bp0']) / 2

        avg_price_to_trade_list = [get_price_avg(observation, vol_to_trade)[0] / mid_price - 1 for vol_to_trade in
                                   [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]

        feature_tensor = torch.tensor([
            observation['code_net_position'] / self.max_position,
            observation['signal0'],
            observation['signal1'],
            observation['signal2'],
            self.relative_time(observation),
            self.price_log(observation),
            self.mid_price_relative(observation),
            # *ask_price_levels,
            # *bid_price_levels,
            *avg_price_to_trade_list,
            ],
            dtype=torch.float32,
        )

        return feature_tensor

    def relative_time(self, observation):
        e_time = int(observation['eventTime']) // 1000
        hours = e_time // 10000
        minutes = (e_time // 100) % 100
        seconds = e_time % 100

        std_time = hours * 3600 + minutes * 60 + seconds
        if e_time > 130000:
            std_time -= 5400

        std_time -= 34200

        return std_time / 14400 * 2 - 1

    def price_log(self, observation):
        open_price = observation['ap0_t0']
        norm_price = np.log(open_price) / np.log(15000)

        return norm_price * 2 - 1

    def mid_price_relative(self, observation):
        return (observation['ap0'] + observation['bp0']) / (2 * observation['ap0_t0']) - 1

    @property
    def feature_names(self):
        return ['pos', 'sig0', 'sig1', 'sig2', 'time', 'logPrice', 'midPrice'] + [f'avgPrice{vol}' for vol in
                                                                                  [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]


class FeatureEngineVersion2(FeatureEngineVersion1):

    def get_input_shape(self):
        return 24

    def get_feature(self, observation) -> torch.Tensor:
        mid_price = (observation['ap0'] + observation['bp0']) / 2

        avg_price_to_trade_list = [get_price_avg(observation, vol_to_trade)[0] / mid_price - 1 for vol_to_trade in
                                   [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]

        feature_tensor = torch.tensor([
            self.relative_time(observation),
            self.price_log(observation),
            self.mid_price_relative(observation),
            observation['code_net_position'] / self.max_position,
            observation['signal0'],
            observation['signal1'],
            observation['signal2'],
            observation['signal0_rank'] * 2 - 1,
            observation['signal1_rank'] * 2 - 1,
            observation['signal2_rank'] * 2 - 1,
            observation['signal0_mean'],
            observation['signal1_mean'],
            observation['signal2_mean'],
            observation['mid_price_std'],
            *avg_price_to_trade_list,
            ],
            dtype=torch.float32,
        )

        return feature_tensor

    @property
    def feature_names(self):
        return ['time', 'logPrice', 'midPrice', 'pos', 'sig0', 'sig1', 'sig2', 'sig0_rank', 'sig1_rank', 'sig2_rank',
                'sig0_avg', 'sig1_avg', 'sig2_avg', 'priceStd'] + [f'avg_price_{vol}' for vol in
                                                                   [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]


class FeatureEngineVersion3(FeatureEngineVersion2):

    def get_input_shape(self):
        return 35

    def get_feature(self, observation) -> torch.Tensor:

        mid_price = (observation['ap0'] + observation['bp0']) / 2
        avgp_margp_tuple_list = [get_price_avg(observation, vol_to_trade) for vol_to_trade in
                                 [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]

        avg_price_to_trade_list = [(tup[0] / mid_price - 1) * 10000 for tup in avgp_margp_tuple_list]
        margin_price_to_trade_list = [(tup[1] / mid_price - 1) * 10000 for tup in avgp_margp_tuple_list]

        feature_tensor = torch.tensor([
            self.relative_time(observation),
            self.price_log(observation),
            self.mid_price_relative(observation),
            self.rule_decision(observation, mid_price),
            observation['code_net_position'] / self.max_position,
            observation['signal0'],
            observation['signal1'],
            observation['signal2'],
            observation['signal0_rank'] * 2 - 1,
            observation['signal1_rank'] * 2 - 1,
            observation['signal2_rank'] * 2 - 1,
            observation['signal0_mean'],
            observation['signal1_mean'],
            observation['signal2_mean'],
            observation['mid_price_std'],
            *margin_price_to_trade_list
             *avg_price_to_trade_list,
            ],
            dtype=torch.float32,
        )

        return feature_tensor

    def rule_decision(self, observation, mid_price):
        price_predict_signal0 = mid_price * (1 + (observation['signal0'] * 0.0001))
        rule_decision = 0.0
        if (observation['signal0'] > 0.8) and price_predict_signal0 >= observation['ap0']:
            rule_decision = 1.0
        elif (observation['signal0'] < -0.8) and price_predict_signal0 <= observation['bp0']:
            rule_decision = -1.0

        return rule_decision

    @property
    def feature_names(self):
        return ['time', 'logPrice', 'midPrice', 'rule_des', 'pos', 'sig0', 'sig1', 'sig2', 'sig0_rank', 'sig1_rank', 'sig2_rank',
                'sig0_avg', 'sig1_avg', 'sig2_avg', 'priceStd'] + [f'marg_price_{vol}' for vol in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]] + [f'avg_price_{vol}' for vol in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]


class FeatureEngineVersion3_Simple(FeatureEngineVersion3):

    def get_input_shape(self):
        return 17

    def get_feature(self, observation) -> torch.Tensor:

        mid_price = (observation['ap0'] + observation['bp0']) / 2

        feature_tensor = torch.tensor([
            observation['full_pos'],
            self.relative_time(observation),
            self.price_log(observation),
            self.mid_price_relative(observation),
            self.rule_decision(observation, mid_price),
            observation['code_net_position'] / self.max_position,
            observation['signal0'],
            observation['signal1'],
            observation['signal2'],
            observation['signal0_rank'] * 2 - 1,
            observation['signal1_rank'] * 2 - 1,
            observation['signal2_rank'] * 2 - 1,
            observation['signal0_mean'],
            observation['signal1_mean'],
            observation['signal2_mean'],
            observation['mid_price_std'],
            (observation['ap0'] / mid_price - 1) * 10000,
            ],
            dtype=torch.float32,
        )

        return feature_tensor

    @property
    def feature_names(self):
        return ['full_pos', 'time', 'logPrice', 'midPrice', 'rule_des', 'pos', 'sig0', 'sig1', 'sig2', 'sig0_rank', 'sig1_rank', 'sig2_rank',
                'sig0_avg', 'sig1_avg', 'sig2_avg', 'priceStd', 'spread', ]


class FeatureEngineVersion4(FeatureEngineVersion3):

    def get_input_shape(self):
        return 7

    def get_feature(self, observation) -> torch.Tensor:

        mid_price = (observation['ap0'] + observation['bp0']) / 2

        feature_tensor = torch.tensor([
            self.price_log(observation),
            # self.mid_price_relative(observation),
            self.rule_decision(observation, mid_price),
            observation['signal0'],
            observation['signal1'],
            observation['signal2'],
            # observation['signal0_rank'] * 2 - 1,
            # observation['signal1_rank'] * 2 - 1,
            # observation['signal2_rank'] * 2 - 1,
            # observation['signal0_mean'],
            # observation['signal1_mean'],
            # observation['signal2_mean'],
            observation['mid_price_std'],
            (observation['ap0'] / mid_price - 1) * 10000,
            ],
            dtype=torch.float32,
        )

        return feature_tensor

    @property
    def feature_names(self):
        return ['logPrice', 'rule_des', 'sig0', 'sig1', 'sig2', 'priceStd', 'spread', ]

FeatureEngineDummy = FeatureEngineVersion2


#####################
# output wrapper
#####################
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
            action = (self.buy_side, self.vol, obs['ap0'])
        elif action_id == 1:
            action = (self.noop_side, 0., 0.)
        elif action_id == 2:
            action = (self.sell_side, self.vol, obs['bp0'])
        else:
            raise ValueError(f'model output should between [0, {self.get_output_shape()})')
        return action

    def select_action(self, observation, model_input: torch.Tensor) -> Tuple[ActionType, torch.tensor, torch.tensor]:

        if observation['eventTime'] > 145500000:
            return (self.noop_side, 0., 0.), None, torch.zeros(3, dtype=torch.float)

        # 0. inference
        with torch.no_grad():
            model_output = self.model(model_input.to(self.device))
        # 1. postprocess output
        # if observation['full_pos'] == 1:
        #     action_id = model_output[..., 1:].argmax(-1).item() + 1
        # elif observation['full_pos'] == -1:
        #     action_id = model_output[..., :-1].argmax(-1).item()
        # else:
        action_id = model_output.argmax(-1).item()
        action = self.action_id_to_action(action_id, observation)

        self._refresh_count += 1
        if self._refresh_count % self.refresh_model_steps == 0:
            self.refresh_model()

        return action, model_input, model_output

    def random_action(self, observation, model_input) -> Tuple[ActionType, torch.tensor, torch.tensor]:

        if observation['eventTime'] > 145500000:
            return (self.noop_side, 0., 0.), None, torch.zeros(3, dtype=torch.float)
        if observation['full_pos'] > 0:
            action_id = random.randrange(1, 3)
        elif observation['full_pos'] < 0:
            action_id = random.randrange(0, 2)
        else:
            action_id = random.randrange(0, 3)
        action = self.action_id_to_action(action_id, observation)
        model_output = torch.zeros(3, dtype=torch.float)
        model_output[action_id] = 1
        return action, model_input, model_output

#####################
# misc
#####################
from dataclasses import dataclass


@dataclass
class ExplicitControlConf:
    signal_risk_thresh: float



def validate_action(obs: Dict, action: ActionType,
                    max_position: int = 300,
                    signal_risk_thresh: float = -float('inf')) -> (ActionType, bool):
    side, vol, price = action

    if side == 0 and obs['signal0'] < signal_risk_thresh:
        return (1, 0, 0.), True
    if side == 2 and obs['signal0'] > -signal_risk_thresh:
        return (1, 0, 0.), True

    # Extract data from the state
    ask_prices = [obs['ap0'], obs['ap1'], obs['ap2'],
                  obs['ap3'], obs['ap4']]
    bid_prices = [obs['bp0'], obs['bp1'], obs['bp2'],
                  obs['bp3'], obs['bp4']]
    ask_volumes = [obs['av0'], obs['av1'], obs['av2'],
                   obs['av3'], obs['av4']]
    bid_volumes = [obs['bv0'], obs['bv1'], obs['bv2'],
                   obs['bv3'], obs['bv4']]
    code_net_position = obs['code_net_position']

    min_position = -max_position

    # Initialize variables
    is_invalid = False

    # Ensure volume is positive
    if vol < 0:
        vol = 0
        is_invalid = True

    # Handle different sides
    if side == 0:  # Buy

        # Check for max position limit
        if code_net_position + vol > max_position:
            vol = max_position - code_net_position
            is_invalid = True

        # Adjust price and volume based on ask prices and volumes
        for i in range(len(ask_prices)):
            if price < ask_prices[i] and vol > sum(ask_volumes[:i]):
                vol = sum(ask_volumes[:i])
                is_invalid = True
                break
            elif price == ask_prices[i]:
                if vol > sum(ask_volumes[:i + 1]):
                    vol = sum(ask_volumes[:i + 1])
                    is_invalid = True
                break
        else:
            if price > ask_prices[-1] and vol > sum(ask_volumes):
                vol = sum(ask_volumes)
                is_invalid = True

    elif side == 2:  # Sell
        # Check for min position limit
        if code_net_position - vol < min_position:
            vol = code_net_position - min_position
            is_invalid = True

        # Adjust price and volume based on bid prices and volumes
        for i in range(len(bid_prices)):
            if price > bid_prices[i] and vol > sum(bid_volumes[:i]):
                vol = sum(bid_volumes[:i])
                is_invalid = True
                break
            elif price == bid_prices[i]:
                if vol > sum(bid_volumes[:i + 1]):
                    vol = sum(bid_volumes[:i + 1])
                    is_invalid = True
                break
        else:
            if price < bid_prices[-1] and vol > sum(bid_volumes):
                vol = sum(bid_volumes)
                is_invalid = True

    if vol == 0:
        side = 1
        price = 0.

    return (side, vol, price), is_invalid

from typing import Dict, List

import numpy as np


def get_rank(data: List, target):
    rank = 0
    for val in data:
        if val < target:
            rank += 1

    return rank


def std(data):
    if len(data) < 2:
        raise ValueError("Standard deviation requires at least two data points")

    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return variance ** 0.5


def get_price_avg(observation: Dict, vol_to_trade: float):
    """
    vol_to_trade > 0 : BUY
    vol_to_trade < 0 : SELL
    """

    abs_vol = abs(vol_to_trade)

    if abs_vol > 10 or vol_to_trade == 0:
        raise ValueError(f"avg_price_to_trade|{vol_to_trade}")

    ask_price_levels = [
        observation['ap0'],
        observation['ap1'],
        observation['ap2'],
        observation['ap3'],
        observation['ap4'],
    ]
    ask_vol_levels = [
        observation['av0'],
        observation['av1'],
        observation['av2'],
        observation['av3'],
        observation['av4'],
    ]
    bid_price_levels = [
        observation['bp0'],
        observation['bp1'],
        observation['bp2'],
        observation['bp3'],
        observation['bp4'],
    ]
    bid_vol_levels = [
        observation['bv0'],
        observation['bv1'],
        observation['bv2'],
        observation['bv3'],
        observation['bv4'],
    ]

    total_cost = 0
    margin_cost = 0
    if vol_to_trade > 0:
        if sum(ask_vol_levels) < 1e-5:
            return 0
        for ap, av in zip(ask_price_levels, ask_vol_levels):
            if vol_to_trade > av:
                vol_to_trade -= av
                total_cost += av * ap
            else:
                total_cost += vol_to_trade * ap
                margin_cost = ap
                break
        else:
            raise ValueError(f"Invalid vol_to_trade|{observation}|{vol_to_trade}")


    elif vol_to_trade < 0:
        vol_to_trade = -vol_to_trade
        if sum(bid_vol_levels) < 1e-5:
            return 0

        for bp, bv in zip(bid_price_levels, bid_vol_levels):
            if vol_to_trade > bv:
                vol_to_trade -= bv
                total_cost += bv * bp
            else:
                total_cost += vol_to_trade * bp
                margin_cost = bp
                break
        else:
            raise ValueError(f"Invalid vol_to_trade|{observation}|{vol_to_trade}")

    return total_cost / abs_vol, margin_cost


def log1p_abs(x):
    if x >= 0:
        return np.log1p(x)
    else:
        return -np.log1p(-x)

#####################
# submission
#####################

max_position = 300

feature_engine = FeatureEngineVersion4(max_position=max_position)
model = DNN(
    input_dim=feature_engine.get_input_shape(),
    output_dim=Action3OutputWrapper.get_output_shape(),
    hidden_dim=[16, 16, 16])
current_fp = os.path.dirname(__file__)
checkpoint = torch.load(os.path.join(current_fp, '1200000.pt'))
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

