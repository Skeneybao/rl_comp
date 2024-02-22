import subprocess
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, Type, Any

import nni
from torch import nn

from training.DQN.actor import ActorConfig
from training.DQN.learner import LearnerConfig
from training.default_param import default_param
from training.model.DNN import DNN
from training.model.QRDNN import QRDNN
from training.model.get_model import get_model
from training.model_io.featureEngine import FeatureEngine
from training.model_io.get_feature_engine import get_feature_engine_type
from training.model_io.output_wrapper import ActionType, get_output_wrapper, ModelOutputWrapper, Action3OutputWrapper, \
    Action3OutputQuantileWrapper
from training.replay.ReplayBuffer import ReplayBuffer
from training.replay.get_replay_buffer import get_replay_buffer
from training.reward.get_reward import get_reward
from training.util.explicit_control import ExplicitControlConf
from training.util.logger import logger


@dataclass
class ExpInfo:
    git_branch: str
    git_commit: str
    git_clean: bool
    nni_exp_id: str
    nni_trial_id: str

@dataclass
class ControlConf:
    training_episode_num: int
    learning_period: int
    nn_init_exist_model: bool = False
    nn_init_add_noise: bool = False
    nn_init_model_path: str = ''

@dataclass
class EnvConf:
    mode: str
    reward_fn: Callable[[int, Dict, Dict, ActionType], float]

def get_git_info() -> (str, str, bool):
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    git_clean = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip() == ''
    return git_branch, git_commit, git_clean


def get_exp_info() -> ExpInfo:
    try:
        git_branch, git_commit, git_clean = get_git_info()
    except subprocess.CalledProcessError:
        git_branch, git_commit, git_clean = 'unknown', 'unknown', False
    nni_exp_id = nni.get_experiment_id()
    nni_trial_id = nni.get_trial_id()
    return ExpInfo(
        git_branch=git_branch,
        git_commit=git_commit,
        git_clean=git_clean,
        nni_exp_id=nni_exp_id,
        nni_trial_id=nni_trial_id,
    )


def get_param_from_nni() -> Tuple[
    ControlConf,
    EnvConf,
    Type[FeatureEngine], Dict[str, Any],
    Type[nn.Module], Dict[str, Any],
    Type[ModelOutputWrapper], Dict[str, Any],
    Type[ReplayBuffer],
    Dict[str, Any],
    ActorConfig,
    LearnerConfig,
    ExplicitControlConf,
]:
    raw_params = nni.get_next_parameter()

    params: Dict[str, Any] = {**default_param, **raw_params}

    # control
    control_param = ControlConf(
        training_episode_num=params['episode_num'],
        learning_period=params['learning_period'],
    )

    # env
    reward_fn = get_reward(params['env$reward_fn'])
    env_param = EnvConf(
        mode=params['env$mode'],
        reward_fn=reward_fn,
    )

    # feature engine
    feature_engine_type = get_feature_engine_type(params['feature_engine_type'])
    feature_engine_param = {k.replace('feature_engine$', ''): v
                            for k, v in params.items() if k.startswith('feature_engine$')}

    # model
    model_type = get_model(params['model_type'])
    model_param = {k.replace('model$', ''): v
                   for k, v in params.items() if k.startswith('model$')}

    # output_wrapper
    output_wrapper_type = get_output_wrapper(params['output_wrapper_type'])
    output_wrapper_param = {k.replace('output_wrapper$', ''): v
                            for k, v in params.items() if k.startswith('output_wrapper$')}

    # replay buffer
    replay_buffer_type = get_replay_buffer(params['replay_buffer_type'])
    replay_buffer_param = {k.replace('replay_buffer$', ''): v
                           for k, v in params.items() if k.startswith('replay_buffer$')}

    # actor config
    actor_config = ActorConfig(
        **{k.replace('actor_config$', ''): v for k, v in params.items() if k.startswith('actor_config$')}
    )

    # learner config
    learner_config = LearnerConfig(
        **{k.replace('learner_config$', ''): v for k, v in params.items() if k.startswith('learner_config$')}
    )

    explicit_config = ExplicitControlConf(
        signal_risk_thresh=params['signal_risk_thresh'],
    )


    # deal with qr-dqn output wrapper
    if params['learner_config$qrdqn']:
        logger.info("qrdqn is enabled")
        if model_type == DNN:
            logger.info("Auto change model type from DNN to QRDNN")
            model_type = QRDNN
        if output_wrapper_type == Action3OutputWrapper:
            logger.info("Auto change output wrapper type from Action3OutputWrapper to Action3OutputQuantileWrapper")
            output_wrapper_type = Action3OutputQuantileWrapper
        params['learner_config$lr'] = 100 * params['learner_config$lr']
        logger.info(f"Auto change learning rate from {params['learner_config$lr'] / 1000} to {params['learner_config$lr']}")

    return (control_param,
            env_param,
            feature_engine_type, feature_engine_param,
            model_type, model_param,
            output_wrapper_type, output_wrapper_param,
            replay_buffer_type,
            replay_buffer_param,
            actor_config,
            learner_config,
            explicit_config,
            )
