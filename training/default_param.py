default_param = {
    ## control
    'episode_num': 10_000,
    'learning_period': 16,

    ## env param
    'env$mode': 'ordered',
    'env$reward_fn': 'normalized_net_return',

    ## feature engine param
    'feature_engine_type': 'version1',
    # 'feature_engine$sample_param': '1'

    ## replay buffer
    'replay_buffer$capacity': 10000,

    ## model param
    'model_type': 'dnn',
    'model$hidden_dim': [64],

    ## output wrapper param
    'output_wrapper_type': 'action11',
    # 'output_wrapper$sample_param': '1'

    ## actor config
    'actor_config$eps_start': 0.9,
    'actor_config$eps_end': 0.05,
    'actor_config$eps_decay': 1e6,

    ## learner config
    'learner_config$batch_size': 128,
    'learner_config$gamma': 0.99,
    'learner_config$tau': 0.005,
    'learner_config$lr': 1e-5,
    'learner_config$optimizer_type': 'SGD',
    'learner_config$model_save_step': 20000,
}