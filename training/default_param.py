default_param = {
    ## control
    'episode_num': 40_000,
    'learning_period': 32,

    ## env param
    'env$mode': 'ordered',
    'env$reward_fn': 'normalized_net_return',

    ## feature engine param
    'feature_engine_type': 'version3Simple',
    'feature_engine$max_position': 10,
    # 'feature_engine$sample_param': '1'

    ## replay buffer
    'replay_buffer_type': 'PrioritizedReplayBuffer',
    'replay_buffer$capacity': 50000,

    ## model param
    'model_type': 'full_pos_dnn',
    'model$hidden_dim': [32, 32],

    ## output wrapper param
    'output_wrapper_type': 'action3',
    # 'output_wrapper$sample_param': '1'

    ## actor config
    'actor_config$eps_start': 0.9,
    'actor_config$eps_end': 0.05,
    'actor_config$eps_decay': 5627937.9401492765,

    ## learner config
    'learner_config$batch_size': 512,
    'learner_config$gamma': 0.99,
    'learner_config$tau': 0.005,
    'learner_config$lr': 5e-8,
    'learner_config$optimizer_type': 'AdamW',
    'learner_config$model_save_step': 20000,
    'learner_config$minimal_buffer_size': 2000,
    'learner_config$l2_reg': 0.0,

    ## explicit control param
    'signal_risk_thresh': -float('inf'),
}