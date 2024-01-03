import time
import uuid

from training.DQN.actor import ActorConfig, Actor
from training.DQN.learner import LearnerConfig, DQNLearner
from training.DQN.model import Action11OutputWrapper
from training.env.featureEngine import FeatureEngineVersion1
from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNN, DNNModelConfig
from training.replay.ReplayBuffer import ReplayBuffer
from training.reward.normalized_net_return import cal_reward
from training.util.logger import logger


def get_new_game():
    return TrainingStockEnv(mode='ordered', reward_fn=cal_reward)


if __name__ == '__main__':
    TRAINING_EPI = 1e6
    LEARNING_FREQUENCY = 64

    exp_name = f'{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}-{uuid.uuid4()}'

    feature_engine = FeatureEngineVersion1()
    model = DNN(DNNModelConfig(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape()))
    model_output_wrapper = Action11OutputWrapper(model)
    replay_buffer = ReplayBuffer(1e5)

    actor_config = ActorConfig(
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1e6,
    )
    actor = Actor(
        get_new_game,
        feature_engine,
        model_output_wrapper,
        replay_buffer,
        actor_config,
    )

    learner_config = LearnerConfig(
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        lr=1e-5,
        optimizer_type='SGD',
        model_save_step=1000,
    )
    learner = DQNLearner(
        learner_config,
        model,
        replay_buffer,
        exp_name,
    )

    env = actor.env

    while env.episode_cnt < TRAINING_EPI:
        actor.step()

        if env.step_cnt % LEARNING_FREQUENCY == 0:
            loss = learner.step()
            logger.info(f"learner stepping, "
                        f"current step count: {env.step_cnt}, "
                        f"current episode count: {env.episode_cnt}, "
                        f"learning loss: {loss}")
