import logging
import sys


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


logger = get_logger()
