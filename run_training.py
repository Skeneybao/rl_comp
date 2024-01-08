import os

import nni

from training.DQN.actor import Actor
from training.DQN.learner import DQNLearner
from training.env.trainingEnv import TrainingStockEnv
from training.replay.ReplayBuffer import ReplayBuffer
from training.util.exp_management import get_exp_info, get_param_from_nni
from training.util.logger import logger

if __name__ == '__main__':
    TRAINING_EPISODE_NUM = 1e6
    LEARNING_PERIOD = 16
    SAVING_PREFIX = '/mnt/data3/rl-data/training_res'

    # Gen exp info & metadata

    exp_info = get_exp_info()
    exp_name = f'{exp_info.nni_exp_id}/{exp_info.nni_trial_id}'

    saving_path = os.path.join(SAVING_PREFIX, exp_info.nni_exp_id, exp_info.nni_trial_id)
    os.makedirs(saving_path)
    with open(os.path.join(saving_path, 'exp_info.txt'), 'w') as f:
        f.write(f'nni_exp_id: {exp_info.nni_exp_id}\n')
        f.write(f'nni_trial_id: {exp_info.nni_trial_id}\n')
        f.write(f'git_branch: {exp_info.git_branch}\n')
        f.write(f'git_commit: {exp_info.git_commit}\n')
        f.write(f'git_clean: {exp_info.git_clean}\n')

    # get params
    (control_param,
     env_param,
     feature_engine_type, feature_engine_param,
     model_type, model_param,
     output_wrapper_type, output_wrapper_param,
     replay_buffer_param,
     actor_config,
     learner_config,
     ) = get_param_from_nni()

    # init
    feature_engine = feature_engine_type(**feature_engine_param)
    model = model_type(input_dim=feature_engine.get_input_shape(), output_dim=output_wrapper_type.get_output_shape(),
                       **model_param)
    model_output_wrapper = output_wrapper_type(model, **output_wrapper_param)
    replay_buffer = ReplayBuffer(**replay_buffer_param)

    # env
    env = TrainingStockEnv(
        mode=env_param.mode,
        reward_fn=env_param.reward_fn,
        save_metric_path=saving_path,
        save_code_metric=True)

    # actor
    actor = Actor(
        env,
        feature_engine,
        model_output_wrapper,
        replay_buffer,
        actor_config,
    )

    # learner
    learner_config.model_save_prefix = SAVING_PREFIX

    learner = DQNLearner(
        learner_config,
        model,
        replay_buffer,
        saving_path,
    )

    # debug

    print("exp_info: ", exp_info)
    print("control_param: ", control_param)
    print("env_param: ", env_param)
    print("feature_engine_type: ", feature_engine_type)
    print("feature_engine_param: ", feature_engine_param)
    print("model_type: ", model_type)
    print("model_param: ", model_param)
    print("output_wrapper_type: ", output_wrapper_type)
    print("output_wrapper_param: ", output_wrapper_param)
    print("replay_buffer_param: ", replay_buffer_param)
    print("actor_config: ", actor_config)
    print("learner_config: ", learner_config)

    loss_acc = []

    while env.episode_cnt < TRAINING_EPISODE_NUM:
        actor.step()

        if env.step_cnt % LEARNING_PERIOD == 0:
            loss = learner.step()
            loss_acc.append(loss)
            if env.step_cnt % (1000 * LEARNING_PERIOD) == 0:
                avg_loss = sum(loss_acc) / len(loss_acc)
                loss_acc = []

                metrics = {'avg_loss': avg_loss}

                nni.report_intermediate_result(metrics)
                logger.info(f"learner stepping, "
                            f"current step count: {env.step_cnt}, "
                            f"current episode count: {env.episode_cnt}, "
                            f"metrics: {metrics}")
