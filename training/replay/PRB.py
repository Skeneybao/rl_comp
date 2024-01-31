import random
from typing import Iterable, Tuple, List, Optional

import numba
import numpy as np

from training.replay.ReplayBuffer import ReplayBuffer
from training.util.sumtree import SumTree


@numba.jit(nopython=True)
def fast_choice(num: int, probs: np.ndarray) -> int:
    x = random.random() * np.sum(probs)
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return i
    return num - 1


@numba.jit(nopython=True)
def fast_choice_multiple(num: int, probs: np.ndarray, size: int) -> List[int]:
    res = []
    s = np.sum(probs)
    for _ in range(size):
        x = random.random() * s
        cum = 0
        i = 0
        for i, p in enumerate(probs):
            cum += p
            if x < cum:
                break
        res.append(i)
    return res


class CircularBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = list()
        self.idx = 0

    def push(self, data):
        if len(self.memory) < self.capacity:
            self.memory.append(data)
        else:
            self.memory[self.idx] = data
            self.idx = (self.idx + 1) % self.capacity

    def append(self, data):
        self.push(data)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]

    def __setitem__(self, idx, value):
        self.memory[idx] = value


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, beta: float = 0.4, alpha: float = 0.6, epsilon: float = 1e-6):
        super().__init__(capacity)
        self.memory = CircularBuffer(capacity)
        self.weight = SumTree(capacity)
        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon

    def push(self, data):
        self.memory.append(data)
        self.weight.add(1.0)

    def sample(self, batch_size: int, replay_by: str = 'random') -> Iterable:
        assert replay_by in ['random']
        if replay_by == 'random':
            return self.sample_random(batch_size)
        else:
            raise NotImplementedError

    def sample_idx(self, total_weight: Optional[float] = None):
        if total_weight is None:
            total_weight = self.weight.total()
        rnd = random.random() * total_weight
        idx, raw_weight = self.weight.get_index_data(rnd)
        return idx, raw_weight

    def sample_random(self, batch_size: int):
        """

        :param batch_size:
        :return: samples: (batch_size,), sample_indices: (batch_size,), loss_weights: (batch_size,)
        """
        # get prioritized experience replay weight
        samples = [self.sample_idx() for _ in range(batch_size)]
        idxs, raw_weights = zip(*samples)
        loss_weights = np.power(len(self.memory) * np.array(raw_weights), -self.beta)
        return [self.memory[i] for i in idxs], idxs, loss_weights

    def sample_batched_ordered(
            self, batch_size: int, batch_length: int
    ) -> Tuple[Iterable[Iterable], Iterable, np.ndarray]:
        """

        :param batch_size:
        :param batch_length:
        :return: samples: (batch_size, batch_length), sample_indices: (batch_size,), loss_weights: (batch_size,)
        """
        idxs = []
        raw_weights = []
        while len(idxs) < batch_size:
            idx, raw_weight = self.sample_idx()
            if idx + batch_length >= len(self.memory):
                continue
            idxs.append(idx)
            raw_weights.append(raw_weight)
        samples = []
        for start_id in idxs:
            sample = [self.memory[i] for i in range(start_id, start_id + batch_length)]
            samples.append(sample)

        loss_weights = np.power(len(self.memory) * np.array(raw_weights), -self.beta)
        return samples, idxs, loss_weights

    def update_weight(self, idx: int, weight: float):
        weight = np.power(weight, self.alpha) + self.epsilon
        self.weight.update(idx, weight)

    def update_weight_batch(self, ids: Iterable[int], weights: Iterable[float]):
        for idx, weight in zip(ids, weights):
            self.update_weight(idx, weight)

    def __len__(self):
        return len(self.memory)
