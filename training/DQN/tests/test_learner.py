import unittest

from env.chooseenv import make
from training.DQN.actor import run_actor, ActorConfig
from training.DQN.counter import Counter
from training.DQN.learner import DQNLearner, LearnerConfig
from training.DQN.model import ActorModel, ActorModelConfig, ModelIOWrapper
from training.replay.ReplayBuffer import ReplayBuffer
from training.reward.dummy_reward import cal_reward as dummy_reward


class LearnerTestCase(unittest.TestCase):
    @staticmethod
    def new_game():
        return make('kafang_stock', seed=None)

    def test_optimize(self):
        model = ActorModel(ActorModelConfig(28, 64, 11))
        model_io_wrapper = ModelIOWrapper(model)
        replay_buffer = ReplayBuffer(1024)
        learner = DQNLearner(
            LearnerConfig(),
            model_io_wrapper,
            replay_buffer
        )
        counter = Counter()

        run_actor(self.new_game, model_io_wrapper, replay_buffer, counter, ActorConfig(0.9, 0.05, 1000),
                  reward_fn=dummy_reward)

        losses = [learner.run_optimize_step() for _ in range(100)]

        self.assertGreater(losses[0], losses[-1])

    def test_update_target_model(self):
        model = ActorModel(ActorModelConfig(28, 64, 11))
        model_io_wrapper = ModelIOWrapper(model)
        replay_buffer = ReplayBuffer(1024)
        learner_config = LearnerConfig()
        learner = DQNLearner(
            learner_config,
            model_io_wrapper,
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
