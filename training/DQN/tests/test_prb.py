import random
import unittest

import numpy as np

from training.DQN.actor import ActorConfig, Actor
from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNN
from training.model_io.featureEngine import FeatureEngineDummy
from training.model_io.output_wrapper import Action11OutputWrapper
from training.replay.PRB import PrioritizedReplayBuffer
from training.util.sumtree import SumTree


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
        replay_buffer = PrioritizedReplayBuffer(1024)

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

    def test_sum_tree(self):
        t = SumTree(10)

        # add
        for i in range(10):
            t.add(i)
        for i in range(10):
            self.assertEqual(t[i], i)

        # update
        for i in range(10):
            t.update(i, i ** 3)
        for i in range(10):
            self.assertEqual(t[i], i ** 3)

        # update batch
        t.batch_update(np.array(range(10)), np.array([i ** 4 for i in range(10)], dtype=np.float64))
        for i in range(10):
            self.assertEqual(t[i], i ** 4)

    def test_sample_batched_ordered(self):
        # fetch
        samples_batches, idxs, loss_weights = self.replay_buffer.sample_batched_ordered(10, 5)
        self.assertEqual(len(samples_batches), 10)
        for samples, idx in zip(samples_batches, idxs):
            self.assertLessEqual(len(samples), 5)
            saved_state = None
            self.assertEqual(self.replay_buffer.memory[idx], samples[0])
            for sample in samples:
                state, model_output, reward, next_state, done = sample
                if saved_state is not None:
                    self.assertTrue((saved_state == state).all())
                saved_state = next_state
        # update
        idx = 42
        self.replay_buffer.update_weight(idx, 100000000)
        samples_batches, idxs, loss_weights = self.replay_buffer.sample_batched_ordered(10, 5)
        self.assertIn(42, idxs)

        # update another
        idx = 43
        self.replay_buffer.update_weight(idx, 1000000000)
        samples_batches, idxs, loss_weights = self.replay_buffer.sample_batched_ordered(10, 5)
        self.assertIn(43, idxs)

        # update all by random
        for i in range(len(self.replay_buffer)):
            self.replay_buffer.update_weight(i, (random.random() * 10) ** 3)
            self.assertGreater(self.replay_buffer.weight.total(), 0)

        for _ in range(10):
            samples_batches, idxs, loss_weights = self.replay_buffer.sample_batched_ordered(100, 5)
            avg_weights = np.mean([self.replay_buffer.weight[i] for i in idxs])
            all_weights = np.mean([self.replay_buffer.weight[i] for i in range(len(self.replay_buffer))])
            self.assertGreater(avg_weights, all_weights)

        # update all by random in batch
        weights = [(random.random() * 10) ** 3 for _ in range(len(self.replay_buffer))]
        self.replay_buffer.update_weight_batch(range(len(self.replay_buffer)), weights)

        for _ in range(10):
            samples_batches, idxs, loss_weights = self.replay_buffer.sample_batched_ordered(100, 5)
            avg_weights = np.mean([self.replay_buffer.weight[i] for i in idxs])
            all_weights = np.mean([self.replay_buffer.weight[i] for i in range(len(self.replay_buffer))])
            self.assertGreater(avg_weights, all_weights)


if __name__ == '__main__':
    unittest.main()
