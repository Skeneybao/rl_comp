import time
import uuid
import os

from training.DQN.actor import ActorConfig, Actor
from training.DQN.learner import LearnerConfig, DQNLearner
from training.model_io.output_wrapper import Action11OutputWrapper, Action3OutputWrapper
from training.model_io.featureEngine import FeatureEngineVersion3_Simple
from training.env.trainingEnv import TrainingStockEnv
from training.model.DNN import DNN
from training.replay.ReplayBuffer import ReplayBuffer
from training.reward.normalized_net_return import cal_reward
from training.util.logger import logger


if __name__ == '__main__':
    TRAINING_EPISODE_NUM = 1e6
    LEARNING_PERIOD = 16
    SAVING_PATH = '/mnt/data3/rl-data/training_res'

    exp_name = f'{time.strftime("%Y%m%d:%H%M%S", time.localtime())}-{str(uuid.uuid4())[:8]}'
    os.makedirs(os.path.join(SAVING_PATH, exp_name))

    feature_engine = FeatureEngineVersion3_Simple(max_position=10)
    model = DNN(input_dim=feature_engine.get_input_shape(), hidden_dim=[64], output_dim=Action3OutputWrapper.get_output_shape())
    model_output_wrapper = Action3OutputWrapper(model)
    replay_buffer = ReplayBuffer(10000)

    env = TrainingStockEnv(
        mode='ordered',
        reward_fn=cal_reward,
        save_metric_path=os.path.join(SAVING_PATH, exp_name),
        save_code_metric=True)

    actor_config = ActorConfig(
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1e6,
    )
    actor = Actor(
        env,
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
        #model_save_prefix=SAVING_PATH,
        model_save_step=20000,
    )
    learner = DQNLearner(
        learner_config,
        model,
        replay_buffer,
        os.path.join(SAVING_PATH, exp_name),
    )

    while env.episode_cnt < TRAINING_EPISODE_NUM:
        actor.step()

        if env.step_cnt % LEARNING_PERIOD == 0:
            loss = learner.step()
            if env.step_cnt % (1000*LEARNING_PERIOD) == 0:
                logger.info(f"learner stepping, "
                            f"current step count: {env.step_cnt}, "
                            f"current episode count: {env.episode_cnt}, "
                            f"learning loss: {loss}")
