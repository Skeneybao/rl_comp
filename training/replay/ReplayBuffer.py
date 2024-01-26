import random
from collections import deque
from typing import Iterable


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, data):
        self.memory.append(data)

    def sample(self, batch_size: int, replay_by: str = 'random') -> Iterable:
        assert replay_by in ['random', 'ordered']
        if replay_by == 'random':
            return self.sample_random(batch_size)
        elif replay_by == 'ordered':
            return self.sample_ordered(batch_size)
        else:
            raise NotImplementedError

    def sample_random(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def sample_ordered(self, batch_size: int):
        # assumes that pushed order is the same as the order of the episodes
        # this may not ture for multi-threaded envs
        if len(self.memory) < batch_size:
            raise ValueError(f'Not enough data in replay buffer: {len(self.memory)} < {batch_size}')
        start_id = random.randint(0, len(self.memory) - batch_size)
        # possibly segmented by a done > 0. That is, one batch can contain multiple episodes.
        return [self.memory[i] for i in range(start_id, start_id + batch_size)]

    def sample_batched_ordered(self, batch_size: int, batch_length: int) -> Iterable[Iterable]:
        samples = []
        for _ in range(batch_size):
            samples.append(self.sample_ordered(batch_length))
        return samples

    def __len__(self):
        return len(self.memory)
