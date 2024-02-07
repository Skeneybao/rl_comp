import logging
import sys
import os
import csv
import numpy as np


def get_logger():
    logger = logging.getLogger('rl-comp')
    logger.setLevel(logging.INFO)

    # Create two handlers: one for stdout and one for stderr
    stdout_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.setLevel(logging.INFO)

    # Formatter for handlers
    formatter = logging.Formatter(
        '%(asctime)s - pid %(process)d - %(processName)s - %(module)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    return logger


def log_states(env, obs, feature_engine, state, reward, action, valid_action, model_output):
    current_code = obs['code']
    if env.save_code_metric and current_code in env.codes_to_log:
        output_file_path = os.path.join(env.save_metric_path, 'code_metric', f"{env.date}_{current_code}_states.csv")
        file_exists = os.path.exists(output_file_path)
        with open(output_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=feature_engine.feature_names + ['price', 'TWAP600', 'reward', 'action', 'Vaction'] + [f'Moutputi{i}' for i in range(len(model_output))])
            if not file_exists:
                csv_writer.writeheader()
            obs_midPrice = (obs['ap0'] + obs['bp0']) / 2
            csv_writer.writerow(dict(zip(feature_engine.feature_names + ['price', 'TWAP600', 'reward', 'action', 'Vaction']+ [f'Moutputi{i}' for i in range(len(model_output))], 
                            np.append(state.numpy(), [obs_midPrice, obs['TWAP600'], reward, action, valid_action, *model_output]))))


logger = get_logger()
