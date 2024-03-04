import abc
from dataclasses import field, dataclass
from typing import Optional

import math
import numpy as np

from training.util.tools import get_rank


class Accumulator(abc.ABC):
    @abc.abstractmethod
    def accumulate(self, data):
        pass

    @abc.abstractmethod
    def get_data(self):
        pass


class RollingAvgAccumulator(Accumulator):
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


class AccumulatingMeanAccumulator(Accumulator):
    starting_point = 200 
    def __init__(self):
        self.partial_sum = 0
        self.count = 0
        self.items = []

    def accumulate(self, data):

        self.count += 1
        if self.count > self.starting_point:
            self.partial_sum += data
            self.items.append(data)

    def get_data(self):
        if self.count <= self.starting_point:
            return 0
        return self.partial_sum / (self.count - self.starting_point)
    

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
    spread_avg: AccumulatingMeanAccumulator = field(default_factory=AccumulatingMeanAccumulator)
    sig0_queue: list = field(default_factory=list)
    sig1_queue: list = field(default_factory=list)
    sig2_queue: list = field(default_factory=list)
    sig0_sum: RollingAvgAccumulator = field(default_factory=RollingAvgAccumulator)
    sig1_sum: RollingAvgAccumulator = field(default_factory=RollingAvgAccumulator)
    sig2_sum: RollingAvgAccumulator = field(default_factory=RollingAvgAccumulator)
    mid_price_std: RollingStdAccumulator = field(default_factory=RollingStdAccumulator)
    code_reward_accum: float = 0
    daily_reward_accum: float = 0

    def log(self, mid_price, spread, sig0, sig1, sig2, reward):
        self.spread_avg.accumulate(spread)
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
            info['spread_avg'] = self.spread_avg.get_data()
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
            info['spread_avg'] = 5e-4

        return info


class EnvInfoAppender:
    def __init__(self, max_position):
        self.info_acc = InfoAccumulator()
        self.max_position = max_position

    def accumulate(self, mid_price, spread, sig0, sig1, sig2, reward):
        self.info_acc.log(mid_price, spread, sig0, sig1, sig2, reward)

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
