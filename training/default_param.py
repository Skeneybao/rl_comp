default_param = {
    ## control
    'episode_num': 40_000,
    'learning_period': 16,

    ## env param
    'env$mode': 'ordered',
    'env$reward_fn': 'log_long_short_sight_return',

    ## feature engine param
    'feature_engine_type': 'version1',
    # 'feature_engine$sample_param': '1'

    ## replay buffer
    'replay_buffer$capacity': 10000,

    ## model param
    'model_type': 'dnn',
    'model$hidden_dim': [64, 64],

    ## output wrapper param
    'output_wrapper_type': 'action11',
    # 'output_wrapper$sample_param': '1'

    ## actor config
    'actor_config$eps_start': 0.9,
    'actor_config$eps_end': 0.05,
    'actor_config$eps_decay': 5627937.9401492765,

    ## learner config
    'learner_config$batch_size': 256,
    'learner_config$gamma': 0.99,
    'learner_config$tau': 0.005,
    'learner_config$lr': 2.1409775642148664e-5,
    'learner_config$optimizer_type': 'AdamW',
    'learner_config$model_save_step': 20000,
    'learner_config$minimal_buffer_size': 2000,
}
