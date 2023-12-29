import os
import random
import sys
from pathlib import Path

import numpy as np

from training.util.logger import get_logger

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

logger = get_logger(__package__)


class TrainingStockEnv(Game):

    def __init__(
            self,
            mode='random',  # ['random', 'ordered']
            save_train_metric=True,
    ):

        super(TrainingStockEnv, self).__init__(
            n_player=1,
            is_obs_continuous=True,
            is_act_continuous=False,
            game_name='stockTrainLoop',
            agent_nums=1,
            obs_type=['vector']
        )

        self._save_train_metric = save_train_metric
        self._parquetFile = ParquetFile()

        dateList = [name for name in os.listdir(TRAIN_DATA_PATH) if
                    os.path.isdir(os.path.join(TRAIN_DATA_PATH, name))]

        if mode == 'random':
            self._dateIter = RandomIterator(dateList)
        elif mode == 'ordered':
            dateList.sort()
            self._dateIter = OrderedIterator(dateList)

        self._current_env = None
        self._step_cnt = 0
        self._step_cnt_except_this_episode = 0
        # init as 0, for the first batch should be episode 1
        self._episode_cnt = 0
        self._train_metric_list = []

    def joint_action_space(self):
        return [self.get_single_action_space(0)]

    def get_single_action_space(self, player_idx):
        return [Discrete(3), Box(low=0, high=100, shape=(1,)), Box(low=2000, high=10000, shape=(1,))]

    def reset(self):

        self._episode_cnt += 1

        try:
            old_data_len = len(self._parquetFile.data)
        except AttributeError:
            old_data_len = 0

        date = next(self._dateIter)
        self._parquetFile.filename = os.path.join(TRAIN_DATA_PATH, date)
        self._parquetFile.load()

        data_df = self._parquetFile.data
        code_list = [float(item) for item in data_df['code'].unique()]

        mock_market_data = MockMarketDataCython(np.array(data_df))
        self._current_env = StockBaseEnvCython(date, code_list, mock_market_data)

        obs, done, info = self._current_env.reset()
        observation = {**obs, **info}

        logger.info(f'reset done, '
                    f'old data length: {old_data_len}, '
                    f'new data length: {len(self._parquetFile.data)}, '
                    f'current step count: {self._step_cnt}, '
                    f'step done in this episode: {self._step_cnt - self._step_cnt_except_this_episode}')

        self._step_cnt_except_this_episode = self._step_cnt

        return observation, 0, 0

    def step(self, action):
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

        reward = self.get_reward()

        if done == 2:
            # current code is done, reset the current env
            logger.debug(f'current code is done, reset the current env, dropped obs is: {obs}')
            obs, _, info = self._current_env.reset()
        elif done == 1:
            # current file is done, reset whole thing
            if self._save_train_metric:
                self._train_metric_list.append(self._current_env.get_backtest_metric)
            obs, _, info = self.reset()

        observation = {**obs, **info}

        return observation, reward, done

    def get_reward(self, all_action=None):
        return 0

    def is_terminal(self):
        return False

    @property
    def episode_cnt(self):
        return self._episode_cnt

    @property
    def step_cnt(self):
        return self._step_cnt


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
