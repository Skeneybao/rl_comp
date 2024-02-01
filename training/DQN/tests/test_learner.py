import unittest

import numpy as np

import training.DQN.learner
from training.DQN.actor import ActorConfig, Actor
from training.DQN.learner import DQNLearner, LearnerConfig
from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNN
from training.model_io.featureEngine import FeatureEngineDummy, FeatureEngineVersion3_Simple
from training.model_io.output_wrapper import Action11OutputWrapper, Action3OutputWrapper
from training.replay.PRB import PrioritizedReplayBuffer
from training.replay.ReplayBuffer import ReplayBuffer


class LearnerTestCase(unittest.TestCase):
    @staticmethod
    def new_game():
        return TrainingStockEnv()

    def test_optimize(self):
        feature_engine = FeatureEngineVersion3_Simple()
        model = DNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[32, 32],
                    output_dim=Action3OutputWrapper.get_output_shape())
        model_output_wrapper = Action3OutputWrapper(model)
        replay_buffer = ReplayBuffer(10000)

        actor_config = ActorConfig(0.9, 0.05, 1000)
        actor = Actor(
            self.new_game(),
            feature_engine,
            model_output_wrapper,
            replay_buffer,
            actor_config,
        )
        # gather training data
        for _ in range(11000):
            actor.step()

        learner_config = LearnerConfig()
        learner = DQNLearner(
            learner_config,
            model,
            replay_buffer,
            training.DQN.learner.NOT_SAVING
        )

        losses = [learner.step() for _ in range(1000)]

        # self.assertGreater(np.mean(losses[50:100]), np.mean(losses[-50:]))

    def test_optimize_prb(self):
        feature_engine = FeatureEngineVersion3_Simple()
        model = DNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[32, 32],
                    output_dim=Action3OutputWrapper.get_output_shape())
        model_output_wrapper = Action3OutputWrapper(model)
        replay_buffer = PrioritizedReplayBuffer(10000)

        actor_config = ActorConfig(0.9, 0.05, 1000)
        actor = Actor(
            self.new_game(),
            feature_engine,
            model_output_wrapper,
            replay_buffer,
            actor_config,
        )
        # gather training data
        for _ in range(11000):
            actor.step()

        learner_config = LearnerConfig(batch_size=256)
        learner = DQNLearner(
            learner_config,
            model,
            replay_buffer,
            training.DQN.learner.NOT_SAVING
        )

        losses = [learner.step() for _ in range(1000)]

        # self.assertGreater(np.mean(losses[50:100]), np.mean(losses[-50:]))

    def test_update_target_model(self):
        feature_engine = FeatureEngineDummy()
        model = DNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[64],
                    output_dim=Action11OutputWrapper.get_output_shape())
        replay_buffer = ReplayBuffer(1024)
        learner_config = LearnerConfig()
        learner = DQNLearner(
            learner_config,
            model,
            replay_buffer,
            training.DQN.learner.NOT_SAVING
        )

        fake_target_dict = learner.model.state_dict()

        for k, v in fake_target_dict.items():
            fake_target_dict[k] = v * 2

        learner.target_model.load_state_dict(fake_target_dict)

        learner.update_target_model()
        for k in fake_target_dict.keys():
            self.assertAlmostEquals(
                learner.target_model.state_dict()[k].sum().item() / learner.model.state_dict()[k].sum().item(),
                2 - learner_config.tau,
                delta=learner_config.tau / 100)


if __name__ == '__main__':
    unittest.main()
