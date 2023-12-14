from typing import Callable, Dict

from env.kafang_stock import KaFangStock
from training.DQN.model import ModelIOWrapper, ActionType
from training.replay.ReplayBuffer import ReplayBuffer
from training.reward import dummy_reward
from training.util.validate_action import validate_action


def run_actor(
        game: KaFangStock,
        model_wrapper: ModelIOWrapper,
        replay_buffer: ReplayBuffer,
        reward_fn: Callable[[Dict, ActionType], float] = dummy_reward.cal_reward
):
    """
    Run an episode of the game, and store the experience in the replay buffer.
    :param game:
    :param model_wrapper:
    :param replay_buffer:
    :param reward_fn:
    """
    state = game.reset_game()
    while not game.done:
        action, model_input, model_output = model_wrapper.wrap_inference_single(state)
        next_state, _, done, _, _ = game.step([action])
        reward = reward_fn(state, action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
