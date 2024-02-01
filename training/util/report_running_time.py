import time
from functools import wraps

from training.util.logger import logger


def report_time(report_steps=10000):
    def decorator(func):
        # Initialize call count and total time
        call_count = 0
        accum_time = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal call_count, accum_time

            # Start high-resolution timing
            start_time = time.perf_counter_ns()

            # Call the function
            result = func(*args, **kwargs)

            # Stop high-resolution timing
            end_time = time.perf_counter_ns()

            # Update call count and total time
            call_count += 1
            accum_time += (end_time - start_time)

            # Report at given steps
            if call_count % report_steps == 0:
                avg_time = accum_time / call_count
                avg_time_ms = avg_time / 1e6
                logger.info(f'Function {func} recent {report_steps} calls average running time: {avg_time_ms:.6f} ms.')
                accum_time = 0
                call_count = 0

            return result

        return wrapper

    return decorator
