import abc
from typing import List

import numpy as np
import torch

from training.util.tools import get_price_avg


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
        return 16

    def get_feature(self, observation) -> torch.Tensor:

        mid_price = (observation['ap0'] + observation['bp0']) / 2

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
            (observation['ap0'] / mid_price - 1) * 10000,
        ],
            dtype=torch.float32,
        )

        return feature_tensor

    @property
    def feature_names(self):
        return ['time', 'logPrice', 'midPrice', 'rule_des', 'pos', 'sig0', 'sig1', 'sig2', 'sig0_rank', 'sig1_rank', 'sig2_rank',
                'sig0_avg', 'sig1_avg', 'sig2_avg', 'priceStd', 'spread', ] 

FeatureEngineDummy = FeatureEngineVersion2
