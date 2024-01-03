import abc

import torch
import numpy as np


class FeatureEngine(abc.ABC):
    @abc.abstractmethod
    def get_input_shape(self):
        pass

    @abc.abstractmethod
    def get_feature(self, observation) -> torch.Tensor:
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

        ask_price_levels = [
            observation['ap0'] / mid_price - 1,
            observation['ap1'] / mid_price - 1,
            observation['ap2'] / mid_price - 1,
            observation['ap3'] / mid_price - 1,
            observation['ap4'] / mid_price - 1,  
        ]
        ask_vol_levels = [
            observation['av0'],
            observation['av1'],
            observation['av2'],
            observation['av3'],
            observation['av4'],
        ]
        bid_price_levels = [
            observation['bp0'] / mid_price - 1,
            observation['bp1'] / mid_price - 1,
            observation['bp2'] / mid_price - 1,
            observation['bp3'] / mid_price - 1,
            observation['bp4'] / mid_price - 1,  
        ]
        bid_vol_levels = [
            observation['bv0'],
            observation['bv1'],
            observation['bv2'],
            observation['bv3'],
            observation['bv4'],
        ]

        avg_price_to_trade_list = [self.avg_price_to_trade(
                        observation, 
                        vol_to_trade,
                        ask_price_levels,
                        ask_vol_levels,
                        bid_price_levels,
                        bid_vol_levels,
        )
        for vol_to_trade in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        ]

        feature_tensor = torch.tensor([
            observation['code_net_position'] / 100,
            observation['signal0'],
            observation['signal1'],
            observation['signal2'],
            self.relative_time(observation),
            self.price_log(observation),
            self.mid_price_relative(observation),
            #*ask_price_levels,
            #*bid_price_levels,
            *avg_price_to_trade_list,
        ])

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

    def avg_price_to_trade(self, 
                           observation, 
                           vol_to_trade: int,
                           ask_price_levels,
                           ask_vol_levels,
                           bid_price_levels,
                           bid_vol_levels,
                           ):
        
        abs_vol = abs(vol_to_trade)

        if abs_vol > 10 or vol_to_trade == 0:
            raise ValueError(f"Feature Error|avg_price_to_trade|{vol_to_trade}|{observation}")

        total_cost = 0        
        if vol_to_trade > 0:
            for ap, av in zip(ask_price_levels, ask_vol_levels):
                if vol_to_trade > av:
                    vol_to_trade -= av
                    total_cost += av * ap
                else:
                    total_cost += vol_to_trade * ap
                    break

        elif vol_to_trade < 0:
            vol_to_trade = -vol_to_trade
            for bp, bv in zip(bid_price_levels, bid_vol_levels):
                if vol_to_trade > bv:
                    vol_to_trade -= bv
                    total_cost += bv * bp
                else:
                    total_cost += vol_to_trade * bp
                    break        

        return total_cost / abs_vol   


                    


