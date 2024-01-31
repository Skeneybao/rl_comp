import random
from typing import Iterable, Tuple, List

import numba
import numpy as np

from training.replay.ReplayBuffer import ReplayBuffer


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
        for i, p in enumerate(probs):
            cum += p
            if x < cum:
                break
        res.append(num - 1)
    return res


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, beta: float = 0.4, alpha: float = 0.6, epsilon: float = 1e-6):
        super().__init__(capacity)
        self.weight = list()
        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon

    def push(self, data):
        self.memory.append(data)
        self.weight.append(1)
        while len(self.weight) > len(self.memory):
            self.weight.pop(0)

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

    def sample_batched_ordered(self, batch_size: int, batch_length: int
                               ) -> Tuple[Iterable[Iterable], Iterable, np.ndarray]:
        """

        :param batch_size:
        :param batch_length:
        :return: samples: (batch_size, batch_length), sample_indices: (batch_size,), loss_weights: (batch_size,)
        """
        samples = []
        weight = np.array(self.weight) + self.epsilon
        prob = np.power(weight, self.alpha) / np.sum(np.power(weight, self.alpha))

        idxs = fast_choice_multiple(len(self.memory) - batch_length + 1,
                                    prob[0:len(self.memory) - batch_length + 1],
                                    batch_size)
        for start_id in idxs:
            sample = [self.memory[i] for i in range(start_id, start_id + batch_length)]
            samples.append(sample)

        loss_weights = np.power(len(self.memory) * prob[idxs], -self.beta)
        return samples, idxs, loss_weights

    def update_weight(self, idx: int, weight: float):
        self.weight[idx] = weight

    def update_weight_batch(self, ids: Iterable[int], weights: Iterable[float]):
        for idx, weight in zip(ids, weights):
            self.update_weight(idx, weight)

    def __len__(self):
        return len(self.memory)
