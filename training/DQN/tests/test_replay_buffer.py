import unittest

from training.DQN.actor import ActorConfig, Actor
from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNN
from training.model_io.featureEngine import FeatureEngineDummy
from training.model_io.output_wrapper import Action11OutputWrapper
from training.replay.ReplayBuffer import ReplayBuffer


class ReplayBufferTestCase(unittest.TestCase):
    @staticmethod
    def new_game():
        return TrainingStockEnv()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        feature_engine = FeatureEngineDummy()
        model = DNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[64],
                    output_dim=Action11OutputWrapper.get_output_shape())
        model_output_wrapper = Action11OutputWrapper(model)
        replay_buffer = ReplayBuffer(1024)

        actor_config = ActorConfig(0.9, 0.05, 1000)
        actor = Actor(
            self.new_game(),
            feature_engine,
            model_output_wrapper,
            replay_buffer,
            actor_config,
        )
        # gather training data
        for _ in range(1024):
            actor.step()

        self.replay_buffer = replay_buffer

    def test_sample_batched_ordered(self):
        for samples in self.replay_buffer.sample_batched_ordered(10, 5):
            self.assertEqual(len(samples), 5)
            saved_state = None
            for sample in samples:
                state, model_output, reward, next_state, done = sample
                if saved_state is not None:
                    self.assertTrue((saved_state == state).all())
                saved_state = next_state


if __name__ == '__main__':
    unittest.main()
