import atexit
import multiprocessing
import os
from collections import deque

import nni

from evaluator import EvaluatorConfig, evaluate_model
from training.DQN.actor import Actor, cal_epsilon
from training.DQN.learner import DQNLearner
from training.env.trainingEnv import TrainingStockEnv
from training.replay.ReplayBuffer import ReplayBuffer
from training.util.exp_management import get_exp_info, get_param_from_nni
from training.util.logger import logger

multiprocessing.set_start_method('spawn', force=True)

# metric used to evaluate model
DEFAULT_METRIC_KEY = 'daily_pnl_mean_sharped'

# second_best or best
FINAL_METRIC_STRATEGY = 'rolling_second_best'


def evaluate_model_process(
        eval_config: EvaluatorConfig,
        avg_loss: float,
        model_name: int,
        result_queue: multiprocessing.Queue):
    metrics = evaluate_model(eval_config)
    metrics['default'] = metrics[DEFAULT_METRIC_KEY]
    result_queue.put((model_name, {**metrics, 'avg_loss': avg_loss, 'model_name': f'{model_name}.pt'}))


if __name__ == '__main__':
    SAVING_PREFIX = '/mnt/data3/rl-data/training_res'

    if os.path.exists('/mnt/data3/rl-data/training_res/STANDALONE/STANDALONE'):
        # recursively remove
        os.system('rm -rf /mnt/data3/rl-data/training_res/STANDALONE/STANDALONE')

    # Gen exp info & metadata
    #################################
    # init exp
    #################################
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

    #################################
    # set object's param on nni's next parameters
    #################################

    # get params
    (control_param,
     env_param,
     feature_engine_type, feature_engine_param,
     model_type, model_param,
     output_wrapper_type, output_wrapper_param,
     replay_buffer_param,
     actor_config,
     learner_config,
     explicit_config,
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
        save_code_metric=True,
        max_postion=feature_engine.max_position)

    # actor
    actor = Actor(
        env,
        feature_engine,
        model_output_wrapper,
        replay_buffer,
        actor_config,
        explicit_config,
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
    logger.warning(f"exp_info: {exp_info}")
    logger.warning(f"control_param: {control_param}")
    logger.warning(f"env_param: {env_param}")
    logger.warning(f"feature_engine_type: {feature_engine_type}")
    logger.warning(f"feature_engine_param: {feature_engine_param}")
    logger.warning(f"model_type: {model_type}")
    logger.warning(f"model_param: {model_param}")
    logger.warning(f"output_wrapper_type: {output_wrapper_type}")
    logger.warning(f"output_wrapper_param: {output_wrapper_param}")
    logger.warning(f"replay_buffer_param: {replay_buffer_param}")
    logger.warning(f"actor_config: {actor_config}")
    logger.warning(f"learner_config: {learner_config}")
    logger.warning(f"explicit_config: {explicit_config}")

    #################################
    # launch exp
    #################################

    loss_acc = []
    latest_model_num = None

    eval_processes = deque()
    result_queue = multiprocessing.Queue()
    result_dict = {}


    def cleanup():
        for process in eval_processes:
            process.terminate()


    atexit.register(cleanup)

    while env.episode_cnt < control_param.training_episode_num:
        actor.step()

        if env.step_cnt % control_param.learning_period == 0:
            loss = learner.step()
            loss_acc.append(loss)
            if env.step_cnt % (1000 * control_param.learning_period) == 0:
                loss_acc = [loss for loss in loss_acc if loss is not None]
                avg_loss = sum(loss_acc) / len(loss_acc)
                loss_acc = []

                should_eval = learner.latest_model_num is not None and latest_model_num != learner.latest_model_num
                if should_eval:
                    #################################
                    # eval in another process
                    #################################
                    latest_model_num = learner.latest_model_num

                    eval_config = EvaluatorConfig(
                        data_path='/home/rl-comp/Git/rl_comp/env/stock_raw/data',
                        date='ALL',
                        training_res_path=saving_path,
                        model_name=f'{latest_model_num}.pt',
                        feature_engine_type=feature_engine_type,
                        feature_engine_param=feature_engine_param,
                        model_type=model_type,
                        model_param=model_param,
                        output_wrapper_type=output_wrapper_type,
                        reward_fn=env_param.reward_fn,
                        explicit_config=explicit_config,
                    )
                    eval_process = multiprocessing.Process(
                        target=evaluate_model_process,
                        args=(eval_config, avg_loss, latest_model_num, result_queue),
                        name=f'eval_{latest_model_num}_process',
                    )
                    eval_process.start()
                    eval_processes.append(eval_process)
                #################################
                # try to report available result
                #################################
                if len(eval_processes) > 0:
                    head_process: multiprocessing.Process = eval_processes[0]
                    if not head_process.is_alive():
                        (key, result) = result_queue.get()
                        result_dict[key] = result
                        nni.report_intermediate_result(result)
                        eval_processes.popleft()

                epsilon = cal_epsilon(actor_config, env.step_cnt)
                logger.info(f"learner stepping, "
                            f"current actor step count: {env.step_cnt}, "
                            f"current learner step count: {learner.step_cnt}, "
                            f"current episode count: {env.episode_cnt}, "
                            f"current epsilon: {epsilon}, "
                            f"avg_loss: {avg_loss}")
    #################################
    # report final result
    #################################

    if latest_model_num != learner.latest_model_num:
        # need to evaluate last model
        eval_config = EvaluatorConfig(
            data_path='/home/rl-comp/Git/rl_comp/env/stock_raw/data',
            date='ALL',
            training_res_path=saving_path,
            model_name=f'{learner.latest_model_num}.pt',
            feature_engine_type=feature_engine_type,
            feature_engine_param=feature_engine_param,
            model_type=model_type,
            model_param=model_param,
            output_wrapper_type=output_wrapper_type,
        )
        eval_process = multiprocessing.Process(
            target=evaluate_model_process,
            args=(eval_config, sum(loss_acc) / len(loss_acc), latest_model_num, result_queue),
            name=f'eval_{latest_model_num}_process',
        )
        eval_process.start()
        eval_processes.append(eval_process)

    # clear unfinished eval process
    for process in eval_processes:
        process.join()
        (key, result) = result_queue.get()
        result_dict[key] = result
        nni.report_intermediate_result(result)

    # find the best metric to report
    if FINAL_METRIC_STRATEGY == 'best':
        best_metric = max(result_dict.values(), key=lambda x: x['default'])
    elif FINAL_METRIC_STRATEGY == 'second_best':
        best_metric = sorted(result_dict.values(), key=lambda x: x['default'])[-2]
    elif FINAL_METRIC_STRATEGY == 'rolling_second_best':
        latest_consider_num = max(100, int(len(result_dict) * 0.1))
        best_metric = sorted(list(result_dict.values())[-latest_consider_num:], key=lambda x: x['default'])[-2]
    else:
        raise ValueError(f'Unknown FINAL_METRIC_STRATEGY: {FINAL_METRIC_STRATEGY}')
    nni.report_final_result(best_metric)
