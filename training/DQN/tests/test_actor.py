import unittest

from env.chooseenv import make
from training.DQN.actor import run_actor
from training.DQN.model import Actor, ActorConfig, ModelIOWrapper
from training.replay.ReplayBuffer import ReplayBuffer
from training.reward.dummy_reward import cal_reward as dummy_reward


class ActorTestCase(unittest.TestCase):
    game = make('kafang_stock', seed=None)

    def test_run_actor(self):
        model = Actor(ActorConfig(28, 64, 11))
        model_io_wrapper = ModelIOWrapper(model)
        replay_buffer = ReplayBuffer(1024)
        run_actor(self.game, model_io_wrapper, replay_buffer, dummy_reward)

        self.assertTrue(0 < len(replay_buffer.memory) <= 1024)
        self.assertTrue(len([1 for _, _, _, _, done in replay_buffer.memory if done != 0]) == 1)


if __name__ == '__main__':
    unittest.main()
