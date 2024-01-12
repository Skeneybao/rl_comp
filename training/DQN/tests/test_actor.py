import unittest

import math

from training.DQN.actor import ActorConfig, if_epsilon_greedy, Actor
from training.model_io.output_wrapper import Action11OutputWrapper
from training.model_io.featureEngine import FeatureEngineDummy
from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNN
from training.replay.ReplayBuffer import ReplayBuffer


class ActorTestCase(unittest.TestCase):

    @staticmethod
    def new_game():
        return TrainingStockEnv()

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
        feature_engine = FeatureEngineDummy()
        model = DNN(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape())
        model_output_wrapper = Action11OutputWrapper(model)
        actor_config = ActorConfig(0.9, 0.05, 1000)
        replay_buffer = ReplayBuffer(1024)
        actor = Actor(
            self.new_game(),
            feature_engine,
            model_output_wrapper,
            replay_buffer,
            actor_config,
        )

        # actor should insert into replay buffer
        for _ in range(10):
            actor.step()
        self.assertTrue(len(replay_buffer.memory) == 10)

        for _ in range(1024 - 10):
            actor.step()
        self.assertTrue(len(replay_buffer.memory) == 1024)

        for _ in range(100):
            actor.step()
        self.assertTrue(len(replay_buffer.memory) == 1024)

        # all random should not fail
        all_random_config = ActorConfig(1, 1, 1)
        actor = Actor(
            self.new_game(),
            feature_engine,
            model_output_wrapper,
            replay_buffer,
            all_random_config,
        )
        for _ in range(1024):
            actor.step()

        # all model should not fail
        all_model_config = ActorConfig(0, 0, 1)
        actor = Actor(
            self.new_game(),
            feature_engine,
            model_output_wrapper,
            replay_buffer,
            all_model_config,
        )
        for _ in range(1024):
            actor.step()


if __name__ == '__main__':
    unittest.main()
