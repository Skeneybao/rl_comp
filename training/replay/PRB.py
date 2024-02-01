import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple, List

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
        self.prefetch_count = 1

    def push(self, data):
        self.memory.append(data)
        self.weight.add(1.0)

    def sample(self, batch_size: int, replay_by: str = 'random') -> Iterable:
        assert replay_by in ['random']
        if replay_by == 'random':
            return self.sample_random_sync(batch_size)
        else:
            raise NotImplementedError

    def sample_idx(self):
        rnd = random.random() * self.weight.total()
        idx, raw_weight = self.weight.get_index_data(rnd)
        return idx, raw_weight

    def batch_sample_idx(self, batch_size: int):
        rnds = np.random.rand(batch_size) * self.weight.total()
        idxs, raw_weights = self.weight.batch_get_index_data(rnds)
        return idxs, raw_weights

    def sample_random_sync(self, batch_size: int):
        """

        :param batch_size:
        :return: samples: (batch_size,), sample_indices: (batch_size,), loss_weights: (batch_size,)
        """
        # get prioritized experience replay weight
        idxs, raw_weights = self.batch_sample_idx(batch_size)
        loss_weights = np.power(len(self.memory) * np.array(raw_weights), -self.beta)
        return [self.memory[i] for i in idxs], idxs, loss_weights

    def sample_batched_ordered_sync(
            self, batch_size: int, batch_length: int
    ) -> Tuple[Iterable[Iterable], Iterable, np.ndarray]:
        """

        :param batch_size:
        :param batch_length:
        :return: samples: (batch_size, batch_length), sample_indices: (batch_size,), loss_weights: (batch_size,)
        """
        idxs, raw_weights = self.batch_sample_idx(batch_size)
        # refetch if start_id + batch_length is out of range
        refetch_idx = [(i, idx) for i, idx in enumerate(idxs) if idx + batch_length > len(self.memory)]

        while len(refetch_idx) > 0:
            idxs_refetch, raw_weights_refetch = self.batch_sample_idx(len(refetch_idx))
            for refresh_i, (i, _) in enumerate(refetch_idx):
                idxs[i] = idxs_refetch[refresh_i]
                raw_weights[i] = raw_weights_refetch[refresh_i]
            refetch_idx = [(i, idx) for i, idx in enumerate(idxs) if idx + batch_length > len(self.memory)]

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
        ids = np.array(ids)
        weight = np.power(np.array(weights), self.alpha) + self.epsilon
        self.weight.batch_update(ids, weight)

    def __len__(self):
        return len(self.memory)

    async def async_prefetch(self, batch_size: int, sample_method, *args):
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as executor:
            futures = [
                loop.run_in_executor(
                    executor, sample_method, batch_size, *args
                )
                for _ in range(self.prefetch_count)
            ]
            for future in asyncio.as_completed(futures):
                yield await future

    def sample_random(self, batch_size: int):
        # return self.async_prefetch(batch_size, self.sample_random_sync)
        return self.sample_random_sync(batch_size)

    def sample_batched_ordered(
            self, batch_size: int, batch_length: int
    ):
        # return self.async_prefetch(batch_size, self.sample_batched_ordered_sync, batch_length)
        return self.sample_batched_ordered_sync(batch_size, batch_length)
