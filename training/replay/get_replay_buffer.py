from typing import Type

from training.replay.ReplayBuffer import ReplayBuffer
from training.replay.PRB import PrioritizedReplayBuffer


def get_replay_buffer(type_name: str) -> Type[ReplayBuffer]:
    if type_name == 'ReplayBuffer':
        return ReplayBuffer
    elif type_name == 'PrioritizedReplayBuffer':
        return PrioritizedReplayBuffer
    else:
        raise ValueError(f'Unknown replay buffer type: {type_name}')