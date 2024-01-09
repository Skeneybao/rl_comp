from typing import Callable, Dict

from training.model_io.output_wrapper import ActionType


def get_reward(fn_name: str) -> Callable[[int, Dict, Dict, ActionType], float]:
    if fn_name == 'dummy':
        from training.reward.dummy_reward import cal_reward
        return cal_reward
    elif fn_name == 'normalized_net_return':
        from training.reward.normalized_net_return import cal_reward
        return cal_reward
    else:
        raise ValueError(f'Unknown reward function name: {fn_name}')
