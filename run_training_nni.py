import atexit
import multiprocessing
import os
from collections import deque

import nni
import torch

from evaluator import EvaluatorConfig, evaluate_model
from training.DQN.actor import Actor, cal_epsilon
from training.DQN.learner import DQNLearner
from training.env.trainingEnv import TrainingStockEnv
from training.util.exp_management import get_exp_info, get_param_from_nni
from training.util.logger import logger

multiprocessing.set_start_method('spawn', force=True)

# metric used to evaluate model
DEFAULT_METRIC_KEY = 'daily_pnl_mean_sharped'

# second_best or best
FINAL_METRIC_STRATEGY = 'rolling_avg'


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

    #################################
    # set object's param on nni's next parameters
    #################################

    # get params
    (control_param,
     env_param,
     feature_engine_type, feature_engine_param,
     model_type, model_param,
     output_wrapper_type, output_wrapper_param,
     replay_buffer_type,
     replay_buffer_param,
     actor_config,
     learner_config,
     explicit_config,
     ) = get_param_from_nni()

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

    # init
    feature_engine = feature_engine_type(**feature_engine_param)
    model = model_type(input_dim=feature_engine.get_input_shape(), output_dim=output_wrapper_type.get_output_shape(),
                       **model_param)


    # load existing model's param
    if control_param.nn_init_exist_model:
        current_weight_abs_sum = sum([param.data.abs().sum() for param in model.parameters()])
        logger.info(f'loading existing model from {control_param.nn_init_model_path}')
        existing_model = model_type(input_dim=feature_engine.get_input_shape(), output_dim=output_wrapper_type.get_output_shape(),
                                    **model_param)
        existing_model.load_state_dict(torch.load(control_param.nn_init_model_path)['model_state_dict'])
        existing_weight_abs_sum = sum([param.data.abs().sum() for param in existing_model.parameters()])

        if control_param.nn_init_add_noise:
            for current_param, existing_param in zip(model.parameters(), existing_model.parameters()):
                current_param.data = existing_param.data * (1 + torch.randn_like(existing_param.data) * 0.1)
        else:
            model.load_state_dict(existing_model.state_dict())
        final_weight_abs_sum = sum([param.data.abs().sum() for param in model.parameters()])

        logger.info(f'current_weight_sum: {current_weight_abs_sum}, existing_weight_sum: {existing_weight_abs_sum}, final_weight_sum: {final_weight_abs_sum}')
        del existing_model

    model_output_wrapper = output_wrapper_type(model, **output_wrapper_param)
    replay_buffer = replay_buffer_type(**replay_buffer_param)

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
    logger.warning(f"replay_buffer_type: {replay_buffer_type}")
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
                avg_loss = sum(loss_acc) / len(loss_acc) if len(loss_acc) > 0 else 0
                loss_acc = []

                should_eval = learner.latest_model_num is not None and latest_model_num != learner.latest_model_num
                if should_eval:
                    #################################
                    # eval in another process
                    #################################
                    latest_model_num = learner.latest_model_num

                    eval_config = EvaluatorConfig(
                        data_path='/mnt/data3/rl-data/train_set_nearest_5sec_val',
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
            data_path='/mnt/data3/rl-data/train_set_nearest_5sec_val',
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
        latest_consider_num = min(10, int(len(result_dict) * 0.2))
        best_metric = sorted(list(result_dict.values())[-latest_consider_num:], key=lambda x: x['default'])[-2]
    elif FINAL_METRIC_STRATEGY == 'rolling_avg':
        latest_consider_num = min(10, int(len(result_dict) * 0.2))
        consideration_list = list(result_dict.values())[-latest_consider_num:]
        rolling_avg_metric = {f'avg_{k}': sum([x[k] for x in consideration_list]) / len(consideration_list)
                              for k in consideration_list[0] if k != 'model_name'}

        best_metric = sorted(consideration_list, key=lambda x: x['default'])[-1]
        for k in best_metric:
            rolling_avg_metric[f'best_{k}'] = best_metric[k]

        all_metric = {**rolling_avg_metric, **best_metric}

        best_metric = all_metric
        best_metric['daily_pnl_mean_sharped'] = best_metric['avg_daily_pnl_mean_sharped']
        best_metric['default'] = best_metric['daily_pnl_mean_sharped']

    else:
        raise ValueError(f'Unknown FINAL_METRIC_STRATEGY: {FINAL_METRIC_STRATEGY}')
    nni.report_final_result(best_metric)
