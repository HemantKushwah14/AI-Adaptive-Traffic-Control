"""Conventional fixed-time baseline: cycles 0->1->2->3 with constant green durations."""
from __future__ import annotations

from typing import Sequence

from config.constants import NUM_PHASES


class FixedTimeController:
    name = "fixed_time"

    def __init__(self, green_times: Sequence[float] = (35, 10, 35, 10)) -> None:
        if len(green_times) != NUM_PHASES or any(g < 3 for g in green_times):
            raise ValueError(f"green_times needs {NUM_PHASES} values, each >= 3 seconds")
        self.green_times = tuple(float(g) for g in green_times)

    def reset(self) -> None:
        pass

    def select_phase(self, env) -> int:
        """Called only while the signal is GREEN. Returns the phase that should be active."""
        sig = env.signal
        if sig.elapsed_green >= self.green_times[sig.active_phase]:
            return (sig.active_phase + 1) % NUM_PHASES
        return sig.active_phase