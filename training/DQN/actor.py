import random
from dataclasses import dataclass
from typing import Callable, Dict

import math

from env.kafang_stock import KaFangStock
from training.DQN.counter import Counter
from training.DQN.model import ModelIOWrapper, ActionType
from training.replay.ReplayBuffer import ReplayBuffer
from training.reward import dummy_reward
from training.util.validate_action import validate_action


@dataclass
class ActorConfig:
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 1000


def if_epsilon_greedy(
        config: ActorConfig,
        steps_done: int,
):
    """
    Decide whether to use epsilon greedy strategy. If return True, use epsilon greedy strategy.
    """
    sample = random.random()
    eps_threshold = config.eps_end + (config.eps_start - config.eps_end) * math.exp(-1. * steps_done / config.eps_decay)
    if sample < eps_threshold:
        return True
    else:
        return False


def run_actor(
        new_game_fn: Callable[[], KaFangStock],
        model_wrapper: ModelIOWrapper,
        replay_buffer: ReplayBuffer,
        counter: Counter,
        config: ActorConfig,
        *,
        reward_fn: Callable[[Dict, ActionType], float] = dummy_reward.cal_reward
):
    """
    Run an episode of the game, and store the experience in the replay buffer.
    """

    game = new_game_fn()

    state = game.reset_game()
    while not game.done:
        if if_epsilon_greedy(config, counter.steps_done):
            action, model_input, model_output = model_wrapper.random_action(state)
        else:
            action, model_input, model_output = model_wrapper.wrap_inference_single(state)
        valid_action, is_invalid = validate_action(state, action)
        next_state, _, done, post_info, _ = game.step([valid_action])
        reward = reward_fn(state, action)
        replay_buffer.push([state, action, reward, next_state, done, model_input, model_output])
        state = next_state
        counter.steps_done += 1
