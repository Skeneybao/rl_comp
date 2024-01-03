import unittest

from training.DQN.actor import ActorConfig, Actor
from training.DQN.learner import DQNLearner, LearnerConfig
from training.DQN.model import Action11OutputWrapper
from training.env.featureEngine import FeatureEngineDummy
from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNNModelConfig, DNN
from training.replay.ReplayBuffer import ReplayBuffer


class LearnerTestCase(unittest.TestCase):
    @staticmethod
    def new_game():
        return TrainingStockEnv()

    def test_optimize(self):
        feature_engine = FeatureEngineDummy()
        model = DNN(DNNModelConfig(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape()))
        model_output_wrapper = Action11OutputWrapper(model)
        replay_buffer = ReplayBuffer(1024)

        actor_config = ActorConfig(0.9, 0.05, 1000)
        actor = Actor(
            self.new_game,
            feature_engine,
            model_output_wrapper,
            replay_buffer,
            actor_config,
        )
        # gather training data
        for _ in range(1024):
            actor.step()

        learner_config = LearnerConfig()
        learner = DQNLearner(
            learner_config,
            model,
            replay_buffer
        )

        losses = [learner.step() for _ in range(100)]

        self.assertGreater(losses[0], losses[-1])

    def test_update_target_model(self):
        feature_engine = FeatureEngineDummy()
        model = DNN(DNNModelConfig(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape()))
        replay_buffer = ReplayBuffer(1024)
        learner_config = LearnerConfig()
        learner = DQNLearner(
            learner_config,
            model,
            replay_buffer
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
