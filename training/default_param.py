default_param = {
    ## control
    'episode_num': 40_000,
    'learning_period': 32,
    'nn_init_exist_model': False,
    'nn_init_add_noise': True,
    'nn_init_model_path': '/mnt/data3/rl-data/training_res/ue235ym1/YHxLF/models/2600000.pt',

    ## env param
    'env$mode': 'ordered',
    'env$reward_fn': 'single600T',

    ## feature engine param
    'feature_engine_type': 'version4',
    'feature_engine$max_position': 10,
    # 'feature_engine$sample_param': '1'

    ## replay buffer
    'replay_buffer_type': 'ReplayBuffer',
    'replay_buffer$capacity': 500000,

    ## model param
    'model_type': 'dnn',
    'model$hidden_dim': [16, 16, 16],

    ## output wrapper param
    'output_wrapper_type': 'action3',
    # 'output_wrapper$sample_param': '1'

    ## actor config
    'actor_config$eps_start': 0.9,
    'actor_config$eps_end': 0.05,
    'actor_config$eps_decay': 10000000.0,
    'actor_config$minimal_buffer_size': 50000,

    ## learner config
    'learner_config$batch_size': 512,
    'learner_config$gamma': 0.99,
    'learner_config$tau': 0.005,
    'learner_config$lr': 5e-7,
    'learner_config$optimizer_type': 'AdamW',
    'learner_config$update_target_model_step': 5000,
    'learner_config$model_save_step': 80000,
    'learner_config$minimal_buffer_size': 50000,
    'learner_config$l2_reg': 0.0,
    'learner_config$cyclic_learning_rate': False,
    'learner_config$reward_steps': 1,
    'learner_config$qrdqn': False,
    'learner_config$loss_beta': 0.001,

    ## explicit control param
    'signal_risk_thresh': -float('inf'),
}