from collections import deque
from typing import Iterable, Tuple

import numpy as np

from training.replay.ReplayBuffer import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, beta: float = 0.4, alpha: float = 0.6, epsilon: float = 1e-6):
        self.memory = deque(maxlen=capacity)
        self.weight = list()
        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon

    def push(self, data):
        self.memory.append(data)
        self.weight.append(1)
        self.weight = self.weight[-len(self.memory):]

    def sample(self, batch_size: int, replay_by: str = 'random') -> Iterable:
        assert replay_by in ['random']
        if replay_by == 'random':
            return self.sample_random(batch_size)
        else:
            raise NotImplementedError

    def sample_random(self, batch_size: int):
        """

        :param batch_size:
        :return: samples: (batch_size,), sample_indices: (batch_size,), loss_weights: (batch_size,)
        """
        # get prioritized experience replay weight
        weight = np.array(self.weight) + self.epsilon
        prob = np.power(weight, self.alpha) / np.sum(np.power(weight, self.alpha))
        idxs = np.random.choice(len(self.memory), batch_size, p=prob)
        loss_weights = np.power(len(self.memory) * prob[idxs], -self.beta)
        return [self.memory[i] for i in idxs], idxs, loss_weights

    def sample_ordered(self, batch_size: int):
        raise NotImplementedError('Ordered sampling is not supported in prioritized replay buffer.')

    def __sample_ordered(self, batch_length: int, prob: np.ndarray):
        """

        :param batch_length:
        :return: samples: (batch_length,), idx
        """
        # assumes that pushed order is the same as the order of the episodes
        # this may not ture for multithreading envs
        if len(self.memory) < batch_length:
            raise ValueError(f'Not enough data in replay buffer: {len(self.memory)} < {batch_length}')
        start_id = np.random.choice(len(self.memory) - batch_length + 1, p=prob[0:len(self.memory) - batch_length + 1])
        # possibly segmented by a done > 0. That is, one batch can contain multiple episodes.
        return [self.memory[i] for i in range(start_id, start_id + batch_length)], start_id

    def sample_batched_ordered(self, batch_size: int, batch_length: int) -> Tuple[Iterable[Iterable], Iterable, Iterable]:
        """

        :param batch_size:
        :param batch_length:
        :return: samples: (batch_size, batch_length), sample_indices: (batch_size,), loss_weights: (batch_size,)
        """
        samples = []
        idx = []
        weight = np.array(self.weight) + self.epsilon
        prob = np.power(weight, self.alpha) / np.sum(np.power(weight, self.alpha))
        for _ in range(batch_size):
            sample, idx = self.sample_ordered(batch_length, prob)
            samples.append(sample)
            idx.append(idx)

        loss_weights = np.power(len(self.memory) * prob[idx], -self.beta)
        return samples, idx, loss_weights

    def __len__(self):
        return len(self.memory)
