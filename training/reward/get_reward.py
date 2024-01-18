from typing import Callable, Dict

from training.model_io.output_wrapper import ActionType

register = {}

NO_NAME = 'NO_NAME'


def register_reward(fn_name: str = NO_NAME):
    def decorator(fn: Callable[[int, Dict, Dict, ActionType], float]):
        if fn_name == 'NO_NAME':
            real_fn_name = fn.__name__
        else:
            real_fn_name = fn_name
        if real_fn_name in register:
            raise ValueError(f'Reward function name {real_fn_name} already exists')
        register[real_fn_name] = fn
        return fn

    return decorator


def get_reward(fn_name: str) -> Callable[[int, Dict, Dict, ActionType], float]:
    try:
        return register[fn_name]
    except KeyError:
        print(register)
        raise ValueError(f'Unknown reward function name: {fn_name}')
