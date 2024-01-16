import abc
from typing import Type, List

import torch
import numpy as np

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
    def feature_names(self, observation) -> List[str]:
        pass


class FeatureEngineExample(FeatureEngine):

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


class FeatureEngineDummy(FeatureEngine):

    def get_input_shape(self):
        return 34

    def get_feature(self, observation) -> torch.Tensor:
        return torch.tensor(list(observation.values()))


class FeatureEngineVersion1(FeatureEngine):

    def get_input_shape(self):
        return 17

    def get_feature(self, observation) -> torch.Tensor:

        mid_price = (observation['ap0'] + observation['bp0']) / 2

        avg_price_to_trade_list = [get_price_avg(observation, vol_to_trade)/mid_price - 1 for vol_to_trade in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5] ]

        feature_tensor = torch.tensor([
            observation['code_net_position'] / 100,
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
        return (observation['ap0'] + observation['bp0']) / ( 2 * observation['ap0_t0']) - 1

    @property
    def feature_names(self):
        return ['pos', 'sig0', 'sig1', 'sig2', 'time', 'logPrice', 'midPrice'] + [f'avgPrice{vol}' for vol in[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]


class FeatureEngineVersion2(FeatureEngineVersion1):
    
    def get_input_shape(self):
        return 24

    def get_feature(self, observation) -> torch.Tensor:

        mid_price = (observation['ap0'] + observation['bp0']) / 2

        avg_price_to_trade_list = [get_price_avg(observation, vol_to_trade)/mid_price for vol_to_trade in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5] ]

        feature_tensor = torch.tensor([
            self.relative_time(observation),
            self.price_log(observation),
            self.mid_price_relative(observation),
            observation['code_net_position'] / 100,
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
        return ['time', 'logPrice', 'midPrice', 'pos', 'sig0', 'sig1', 'sig2', 'sig0_rank', 'sig1_rank', 'sig2_rank', 'sig0_avg', 'sig1_avg', 'sig2_avg', 'priceStd'] + [f'avg_price_{vol}' for vol in[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]


