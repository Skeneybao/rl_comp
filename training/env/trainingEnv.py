import os
import random
import sys
import csv
from pathlib import Path
from typing import Callable, Dict

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from training.model_io.output_wrapper import ActionType
from training.util.logger import logger

CURRENT_PATH = str(Path(__file__).resolve().parent.parent)
stock_path = os.path.join(CURRENT_PATH, 'env/stock_raw')
sys.path.append(stock_path)

TRAINING_RAW_DATA = '/mnt/data3/rl-data/train_set'
TRAINING_DATA_5SEC = '/mnt/data3/rl-data/train_set_nearest_5sec/'
TRAIN_DATA_PATH = TRAINING_DATA_5SEC
CODE_TO_PLOT = [486.0, 218.0, 143.0, 492.0]

from env.utils.box import Box
from env.utils.discrete import Discrete
from env.simulators.game import Game
from env.stock_raw.backtest.utils import ParquetFile
from env.stock_raw.mock_market_common.mock_market_data_cython import MockMarketDataCython
from env.stock_raw.envs.stock_base_env_cython import StockBaseEnvCython
from env.stock_raw.utils import Order
from training.reward.dummy_reward import cal_reward as dummy_reward
from training.util.tools import get_rank


class TrainingStockEnv(Game):

    def __init__(
            self,
            mode='random',  # ['random', 'ordered']
            save_metric_path=None,
            save_daily_metric=False,
            save_code_metric=False,
            reward_fn: Callable[[int, Dict, Dict, ActionType], float] = dummy_reward,
            data_path=TRAIN_DATA_PATH,
            dates = 'ALL',
    ):

        super(TrainingStockEnv, self).__init__(
            n_player=1,
            is_obs_continuous=True,
            is_act_continuous=False,
            game_name='stockTrainLoop',
            agent_nums=1,
            obs_type=['vector']
        )

        self.save_metric_path = save_metric_path
        self.save_code_metric = save_code_metric
        self._save_daily_metric = save_daily_metric
        if save_daily_metric:
            with open(os.path.join(self.save_metric_path, "daily_metric.csv"), 'w', newline='') as f:
                self._daily_metric_writer = csv.DictWriter(f, fieldnames=['date', 'code_nums', 'day_pnl', 'daily_return', 'day_handling_fee', 'day_total_orders_num', 'day_total_orders_volume'])
                self._daily_metric_writer.writeheader()
        if save_code_metric and not os.path.exists(os.path.join(save_metric_path, 'code_metric')):
            os.makedirs(os.path.join(save_metric_path, 'code_metric'))

        self.reward_fn = reward_fn
        self._parquetFile = ParquetFile()
        self._data_path = data_path

        if dates == 'ALL':
            dateList = [name for name in os.listdir(self._data_path) if
                        os.path.isdir(os.path.join(self._data_path, name))]
        else:
            dateList = dates

        if mode == 'random':
            self._dateIter = RandomIterator(dateList)
        elif mode == 'ordered':
            dateList.sort()
            self._dateIter = OrderedIterator(dateList)

        self.codes_to_log = CODE_TO_PLOT 
        self._code_pos_path = []
        self._code_price_path = []
        self._code_reward_accum_path = []
        self._mid_price_queue = []
        self._sig0_queue = []
        self._sig1_queue = []
        self._sig2_queue = []
        self._code_reward_accum = 0
        self._daily_reward_accum = 0
        self._current_env = None
        self._step_cnt = 0
        self._step_cnt_except_this_episode = 0
        # init as 0, for the first batch should be episode 1
        self._episode_cnt = 0
        self._reset_cnt = 0

        self._last_obs = None

    def reset(self):
        try:
            old_data_len = len(self._parquetFile.data)
        except AttributeError:
            old_data_len = 0

        date = next(self._dateIter)
        self._parquetFile.filename = os.path.join(self._data_path, date)
        self._parquetFile.load()

        data_df = self._parquetFile.data
        code_list = [float(item) for item in data_df['code'].unique()]

        mock_market_data = MockMarketDataCython(np.array(data_df))
        self._current_env = StockBaseEnvCython(date, code_list, mock_market_data)

        obs, done, info = self._current_env.reset()
        observation = {**obs, **info}
        self._last_obs = observation

        logger.info(f'reset done, '
                    f'old data length: {old_data_len}, '
                    f'new data length: {len(self._parquetFile.data)}, '
                    f'new data code num: {len(code_list)}, '
                    f'current step count: {self._step_cnt}, '
                    f'step done in this episode: {self._step_cnt - self._step_cnt_except_this_episode}')

        if self.save_code_metric:
            with open(os.path.join(self.save_metric_path, 'code_metric',f"{date}.csv"), 'a', newline='') as f:
                code_metric_writer = csv.DictWriter(f, fieldnames=['code', 'time', 'code_net_position', 'code_pnl', 'code_cash_pnl', 'code_positional_pnl', 'code_handling_fee', 'ap0_t0', 'reward'])
                code_metric_writer.writeheader()

        self._reset_cnt += 1
        return observation, 0, info

    def step(self, action: ActionType):
        """
        Action format:
        [side, volume, price]
        """

        self._step_cnt += 1

        order = Order(*action)

        try:
            obs, done, info = self._current_env.step(order)
        except ValueError as v:
            raise ValueError(f'Current game terminate early', v)

        # Record all signal values for current code
        self._mid_price_queue.append((obs['ap0'] + obs['bp0']) / 2)
        self._sig0_queue.append(obs['signal0'])
        self._sig1_queue.append(obs['signal1'])
        self._sig2_queue.append(obs['signal2'])
        
        # Get reward
        # Note that if the game is done in this step, the reward should be calculated based on the last observation of
        # this episode.
        # However, based on the handling of done in the following code, the returned observation is the first one of
        # the next episode, so we use the last observation of the current episode here to calculate the reward.
        reward = self.get_reward(
            self._step_cnt - self._step_cnt_except_this_episode,
            self._last_obs,
            {**obs, **info},
            action,
            
        )

        self._code_reward_accum += reward
        self._daily_reward_accum += reward
        if obs['code'] in self.codes_to_log:
            self._code_pos_path.append(obs['code_net_position'])
            self._code_reward_accum_path.append(self._code_reward_accum)
            self._code_price_path.append((obs['ap0'] + obs['bp0'])/2/obs['ap0_t0'])

        # Handling when done:
        # 0: not done, 1: done in this file, 2: done for this code of stock
        # when done, drop the last observation, and reset the current env, finally return the next observation
        # that is, last_obs = <second last observation of the current file or current code of stock>
        #          obs = <first observation of the next file or next code of stock>
        if done == 2:
            # current code is done, reset the current env
            obs, _, info = self.reset_code(obs, info)
        
        elif done == 1:
            # current file is done, reset whole thing
            if self._save_daily_metric:
                with open(os.path.join(self.save_metric_path, "daily_metric.csv"), 'a', newline='') as f:
                    daily_metric_writer = csv.DictWriter(f, fieldnames=['date', 'code_nums', 'day_pnl', 'daily_return', 'day_handling_fee', 'day_total_orders_num', 'day_total_orders_volume'])
                    daily_metric_writer.writerow(self._current_env.get_backtest_metric())
            self._deal_code_plot(obs['code'])
            
            obs, _, info = self.reset()
            self._step_cnt_except_this_episode = self._step_cnt
            self._episode_cnt += 1
            self._code_reward_accum = 0
            self._daily_reward_accum = 0

            self._code_pos_path = []
            self._code_price_path = []
            self._code_reward_accum_path = []
            self._mid_price_queue = []
            self._sig0_queue = []
            self._sig1_queue = []
            self._sig2_queue = []

        if len(self._sig0_queue) > 240:
            info['signal0_rank'] = get_rank(self._sig0_queue[-240:], obs['signal0']) / 240
            info['signal1_rank'] = get_rank(self._sig1_queue[-240:], obs['signal1']) / 240
            info['signal2_rank'] = get_rank(self._sig2_queue[-240:], obs['signal2']) / 240
            info['signal0_mean'] = np.mean(self._sig0_queue)
            info['signal1_mean'] = np.mean(self._sig1_queue)
            info['signal2_mean'] = np.mean(self._sig2_queue)
            info['mid_price_std'] = np.std(np.array(self._mid_price_queue[-240:])/obs['ap0_t0'])
        else:
            info['signal0_rank'] = 0.5
            info['signal1_rank'] = 0.5
            info['signal2_rank'] = 0.5
            info['signal0_mean'] = 0
            info['signal1_mean'] = 0
            info['signal2_mean'] = 0
            info['mid_price_std'] = 1

        observation = {**obs, **info}
        self._last_obs = observation

        return observation, reward, done

    def reset_code(self, obs, info):
            
            logger.debug(f'current code is done, reset the current env,'
                         f'current step count: {self._step_cnt}, '
                         f'step done in this episode: {self._step_cnt - self._step_cnt_except_this_episode}')
            
            if self.save_code_metric:
                metric_to_log = {
                    'code': obs['code'],
                    'time': obs['eventTime'],
                    'code_net_position': info['code_net_position'], 
                    'code_pnl': info['code_pnl'],
                    'code_cash_pnl': info['code_cash_pnl'], 
                    'code_positional_pnl': info['code_positional_pnl'], 
                    'code_handling_fee': info['code_handling_fee'], 
                    'ap0_t0': info['ap0_t0'],
                    'reward': self._code_reward_accum,
                }
                with open(os.path.join(self.save_metric_path, 'code_metric',f"{self._current_env.date}.csv"), 'a', newline='') as f:
                    code_metric_writer = csv.DictWriter(f, fieldnames=['code', 'time', 'code_net_position', 'code_pnl', 'code_cash_pnl', 'code_positional_pnl', 'code_handling_fee', 'ap0_t0', 'reward'])
                    code_metric_writer.writerow(metric_to_log)

                self._deal_code_plot(obs['code'])

            obs, _, info = self._current_env.reset()
            self._step_cnt_except_this_episode = self._step_cnt
            self._episode_cnt += 1
            self._code_reward_accum = 0

            self._code_pos_path = []
            self._code_price_path = []
            self._code_reward_accum_path = []
            self._sig0_queue = []
            self._sig1_queue = []
            self._sig2_queue = []

            return obs, None, info


        
    def _deal_code_plot(self, code):
        if code in self.codes_to_log:
            fig, ax = plt.subplots()
            ax.plot(np.array(self._code_pos_path)/300, label='net_position')
            ax.plot( (np.array(self._code_price_path) - 1) * 10, label = 'price')
            ax.set_ylim(-1.0, 1.0)
            ax2 = ax.twinx()
            ax2.plot( self._code_reward_accum_path, label='reward_accum', color='plum')
            fig.legend(loc='upper left')
            fig.savefig(os.path.join(self.save_metric_path, 'code_metric',f"{self._current_env.date}_{int(code)}.png"))

    def get_reward(self, step_this_episode: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
        assert obs_after['code'] == obs_before['code']
        return self.reward_fn(step_this_episode, obs_before, obs_after, action)

    def is_terminal(self):
        return False

    def __len__(self):
        return len(self._dateIter)

    def compute_final_stats(self):
        df = pd.read_csv(os.path.join(self.save_metric_path, "daily_metric.csv"))
        stats = {}
        is_traded_day = df['day_total_orders_volume'] != 0
        days_traded = df.loc[is_traded_day, 'day_pnl'].count()
        days_win = sum(df.loc[:, 'day_pnl'] > 0)

        stats['day_pnl_mean'] = df.loc[:, 'day_pnl'].mean()
        stats['daily_return_mean'] = df.loc[:, 'daily_return'].mean()
        stats['code_nums_mean'] = df.loc[:, 'code_nums'].mean()
        stats['day_total_orders_volume_mean'] = df.loc[:,
                                                'day_total_orders_volume'].mean()

        has_traded = days_traded != 0
        stats['win_rate'] = days_win / float(days_traded) if has_traded else 0
        pnl_total_sum = df.loc[:, 'day_pnl'].sum()
        stats['day_traded_pnl_mean'] = pnl_total_sum / \
                                       days_traded if has_traded else 0
        pnl_std = df.loc[:, 'day_pnl'].std(ddof=0)

        std_net_pnl = df.loc[:, 'day_pnl'].std(ddof=0)
        std_net_pnl_notnan = not math.isnan(std_net_pnl)
        std_net_pnl_is_valid = std_net_pnl_notnan and has_traded and std_net_pnl != 0
        stats['sharpe'] = math.sqrt(
            250) * stats['day_pnl_mean'] / pnl_std if std_net_pnl_is_valid else 1

        fee_sum = df.loc[:, 'day_handling_fee'].sum()
        stats['day_handling_fee_mean'] = fee_sum / \
                                         days_traded if has_traded else 0
        if stats['day_pnl_mean'] >= 0:
            stats['daily_return_mean_sharped'] = stats['daily_return_mean'] * \
                                                 min(10, stats['sharpe']) / 10
            stats['daily_pnl_mean_sharped'] = stats['day_pnl_mean'] * \
                                              min(10, stats['sharpe']) / 10
        else:
            stats['daily_return_mean_sharped'] = stats['daily_return_mean']
            stats['daily_pnl_mean_sharped'] = stats['day_pnl_mean']

        return stats

    @property
    def episode_cnt(self):
        return self._episode_cnt

    @property
    def step_cnt(self):
        return self._step_cnt

    @property
    def reset_cnt(self):
        return self._reset_cnt
    
    @property
    def date(self):
        return self._current_env.date
    

class RandomIterator:
    def __init__(self, data):
        self.data = list(data)
        self.length = len(self.data)
        self.index = 0
        self.shuffle_data()

    def shuffle_data(self):
        random.shuffle(self.data)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.length:
            self.shuffle_data()
        result = self.data[self.index]
        self.index += 1
        return result

    def __len__(self):
        return len(self.data)
    

class OrderedIterator:
    def __init__(self, data):
        self.data = list(data)
        self.length = len(self.data)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        result = self.data[self.index]
        self.index = (self.index + 1) % self.length
        return result

    def __len__(self):
        return len(self.data)
