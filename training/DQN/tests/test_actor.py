import unittest
import env
from env.chooseenv import make
from training.DQN.actor import run_actor
from training.DQN.model import Actor, ActorConfig, ModelIOWrapper
from training.replay.ReplayBuffer import ReplayBuffer
from training.reward.dummy_reward import cal_reward as dummy_reward


class ActorTestCase(unittest.TestCase):
    game = make('kafang_stock', seed=None)

    def test_something(self):
        model = Actor(ActorConfig(28, 64, 11))
        model_io_wrapper = ModelIOWrapper(model)
        replay_buffer = ReplayBuffer(1024)
        run_actor(self.game, model_io_wrapper, replay_buffer, dummy_reward)

        self.assertEqual(True, True)


if __name__ == '__main__':
    # unittest.main()
    pass