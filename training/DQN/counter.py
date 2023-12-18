from dataclasses import dataclass


@dataclass
class Counter:
    steps_done: int = 0
