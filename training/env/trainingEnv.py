import os
import random
import sys
import csv
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from training.DQN.model import ActionType
from training.util.logger import logger

CURRENT_PATH = str(Path(__file__).resolve().parent.parent)
stock_path = os.path.join(CURRENT_PATH, 'env/stock_raw')
sys.path.append(stock_path)

TRAINING_RAW_DATA = '/mnt/data3/rl-data/train_set'
TRAINING_DATA_5SEC = '/mnt/data3/rl-data/train_set_nearest_5sec/'

TRAIN_DATA_PATH = TRAINING_DATA_5SEC

from env.utils.box import Box
from env.utils.discrete import Discrete
from env.simulators.game import Game
from env.stock_raw.backtest.utils import ParquetFile
from env.stock_raw.mock_market_common.mock_market_data_cython import MockMarketDataCython
from env.stock_raw.envs.stock_base_env_cython import StockBaseEnvCython
from env.stock_raw.utils import Order
from training.reward.dummy_reward import cal_reward as dummy_reward


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

        self._save_metric_path = save_metric_path
        self._save_code_metric = save_code_metric
        self._save_daily_metric = save_daily_metric
        if save_daily_metric:
            with open(os.path.join(self._save_metric_path, "daily_metric.csv"), 'w', newline='') as f:
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

        if self._save_code_metric:
            with open(os.path.join(self._save_metric_path, 'code_metric',f"{date}.csv"), 'a', newline='') as f:
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

        # Get reward
        # Note that if the game is done in this step, the reward should be calculated based on the last observation of
        # this episode.
        # However, based on the handling of done in the following code, the returned observation is the first one of
        # the next episode, so we use the last observation of the current episode here to calculate the reward.
        reward = self.get_reward(
            self._step_cnt - self._step_cnt_except_this_episode,
            self._last_obs,
            {**obs, **info},
            action
        )

        self._code_reward_accum += reward
        self._daily_reward_accum += reward

        # Handling when done:
        # 0: not done, 1: done in this file, 2: done for this code of stock
        # when done, drop the last observation, and reset the current env, finally return the next observation
        # that is, last_obs = <second last observation of the current file or current code of stock>
        #          obs = <first observation of the next file or next code of stock>
        if done == 2:
            # current code is done, reset the current env
            logger.debug(f'current code is done, reset the current env,'
                         f'current step count: {self._step_cnt}, '
                         f'step done in this episode: {self._step_cnt - self._step_cnt_except_this_episode}')
            
            if self._save_code_metric:
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
                with open(os.path.join(self._save_metric_path, 'code_metric',f"{self._current_env.date}.csv"), 'a', newline='') as f:
                    code_metric_writer = csv.DictWriter(f, fieldnames=['code', 'time', 'code_net_position', 'code_pnl', 'code_cash_pnl', 'code_positional_pnl', 'code_handling_fee', 'ap0_t0', 'reward'])
                    code_metric_writer.writerow(metric_to_log)
            obs, _, info = self._current_env.reset()
            self._step_cnt_except_this_episode = self._step_cnt
            self._episode_cnt += 1
            self._code_reward_accum = 0
        elif done == 1:
            # current file is done, reset whole thing
            if self._save_daily_metric:
                with open(os.path.join(self._save_metric_path, "daily_metric.csv"), 'a', newline='') as f:
                    daily_metric_writer = csv.DictWriter(f, fieldnames=['date', 'code_nums', 'day_pnl', 'daily_return', 'day_handling_fee', 'day_total_orders_num', 'day_total_orders_volume'])
                    daily_metric_writer.writerow(self._current_env.get_backtest_metric())

            obs, _, info = self.reset()
            self._step_cnt_except_this_episode = self._step_cnt
            self._episode_cnt += 1
            self._code_reward_accum = 0
            self._daily_reward_accum = 0

        observation = {**obs, **info}
        self._last_obs = observation

        return observation, reward, done

    def get_reward(self, step_this_episode: int, obs_before: Dict, obs_after: Dict, action: ActionType) -> float:
        assert obs_after['code'] == obs_before['code']
        return self.reward_fn(step_this_episode, obs_before, obs_after, action)

    def is_terminal(self):
        return False

    def __len__(self):
        return len(self._dateIter)
    
    def __del__(self):
        try:
            self._code_output_file.close()
        except:
            pass
        try:
            self._daily_output_file.close()
        except:
            pass

    @property
    def episode_cnt(self):
        return self._episode_cnt

    @property
    def step_cnt(self):
        return self._step_cnt

    @property
    def reset_cnt(self):
        return self._reset_cnt
    

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
