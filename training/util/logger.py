import logging
import sys


def get_logger():
    logger = logging.getLogger('rl-comp')
    logger.setLevel(logging.DEBUG)  # Capture all levels of log

    # Create two handlers: one for stdout and one for stderr
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)  # Log WARNING, ERROR, and CRITICAL to stderr

    # Formatter for handlers
    formatter = logging.Formatter('%(asctime)s - pid %(process)d - %(module)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    # logger.addHandler(stderr_handler)
    return logger


logger = get_logger()
