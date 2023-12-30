from training.DQN.actor import ActorConfig, Actor
from training.DQN.learner import LearnerConfig, DQNLearner
from training.DQN.model import Action11OutputWrapper
from training.env.featureEngine import FeatureEngineDummy
from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNN, DNNModelConfig
from training.replay.ReplayBuffer import ReplayBuffer
from training.util.logger import logger


def get_new_game():
    return TrainingStockEnv(mode='ordered')


if __name__ == '__main__':
    TRAINING_EPI = 1000
    LEARNING_FREQUENCY = 16

    feature_engine = FeatureEngineDummy()
    model = DNN(DNNModelConfig(feature_engine.get_input_shape(), [64], Action11OutputWrapper.get_output_shape()))
    model_output_wrapper = Action11OutputWrapper(model)
    replay_buffer = ReplayBuffer(1024)

    actor_config = ActorConfig(0.9, 0.05, 1000)
    actor = Actor(
        get_new_game,
        feature_engine,
        model_output_wrapper,
        model,
        replay_buffer,
        actor_config,
    )

    learner_config = LearnerConfig()
    learner = DQNLearner(
        learner_config,
        model,
        replay_buffer
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
