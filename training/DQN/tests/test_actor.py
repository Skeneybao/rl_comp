import unittest

import math

from env.chooseenv import make
from training.DQN.actor import run_actor, ActorConfig, if_epsilon_greedy
from training.DQN.counter import Counter
from training.DQN.model import Model, ModelConfig, Actor
from training.replay.ReplayBuffer import ReplayBuffer
from training.reward.dummy_reward import cal_reward as dummy_reward


class ActorTestCase(unittest.TestCase):

    @staticmethod
    def new_game():
        return make('kafang_stock', seed=None)

    def test_epsilon_greedy(self):
        config = ActorConfig(0.9, 0.05, 1000)

        exp_prob = [
            (0, 0.9),
            (1000, 0.05 + (0.9 - 0.05) / math.exp(1)),
            (100000000, 0.05),
        ]
        for steps_done, expected_prob in exp_prob:
            exp = [if_epsilon_greedy(config, steps_done) for _ in range(100000)]
            prob = sum(exp) / len(exp)
            self.assertAlmostEqual(prob, expected_prob, delta=0.005)

    def test_run_actor(self):
        model = Model(ModelConfig(28, 64, 11))
        actor_config = ActorConfig(0.9, 0.05, 1000)
        model_io_wrapper = Actor(model)
        replay_buffer = ReplayBuffer(1024)
        counter = Counter()
        run_actor(self.new_game, model_io_wrapper, replay_buffer, counter, actor_config, reward_fn=dummy_reward)

        self.assertTrue(0 < len(replay_buffer.memory) <= 1024)
        self.assertTrue(len([1 for _, _, _, _, done, _, _ in replay_buffer.memory if done == 2]) == 1)

        all_random_config = ActorConfig(1, 1, 1)
        run_actor(self.new_game, model_io_wrapper, replay_buffer, counter, all_random_config, reward_fn=dummy_reward)

        all_model_config = ActorConfig(0, 0, 1)
        run_actor(self.new_game, model_io_wrapper, replay_buffer, counter, all_model_config, reward_fn=dummy_reward)


if __name__ == '__main__':
    unittest.main()
