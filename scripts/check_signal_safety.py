"""Checks that the DQN-driven signal is safe on every simulation step.
    python scripts\\check_signal_safety.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from config.settings import SimulationConfig  # noqa: E402
from simulation.adaptive_controller import AdaptiveController  # noqa: E402
from simulation.sumo_environment import SumoEnvironment  # noqa: E402
from simulation.traci_controller import SignalMode  # noqa: E402


def main() -> int:
    ctrl = AdaptiveController.from_checkpoint()
    cfg = SimulationConfig.with_uniform_demand(450, duration=300, seed=2)
    bad, changes, prev = [], [0.0], [0]

    def check(env) -> None:
        sig = env.signal
        state = sig.current_state
        greens = {sig._signals[i][0] for i, ch in enumerate(state) if ch == "G"}
        if sig.mode != SignalMode.GREEN and "G" in state:
            bad.append((env.sim_time, "green during transition"))
        if not (greens <= {"N", "S"} or greens <= {"E", "W"}):
            bad.append((env.sim_time, f"conflicting greens {greens}"))
        if sig.mode == SignalMode.GREEN and sig.active_phase != prev[0]:
            changes.append(env.sim_time)
            prev[0] = sig.active_phase

    with SumoEnvironment(cfg, label="p5_safety") as env:
        env.reset()
        env.run(ctrl, on_step=check)

    gaps = np.diff(changes)
    print("controller stats:", ctrl.stats())
    print(f"safety violations: {bad}")
    print(f"phase changes: {len(changes) - 1}")
    lo = cfg.min_green + cfg.yellow_time
    hi = cfg.max_green + cfg.yellow_time + cfg.all_red_time + cfg.decision_interval + 2
    ok = not bad and len(gaps) > 0 and gaps.min() >= lo and gaps.max() <= hi
    if len(gaps):
        print(f"green-to-green gaps: min {gaps.min():.0f}s (must be >= {lo}), max {gaps.max():.0f}s (must be <= {hi})")
    print("[ OK ] signal safety and green limits hold" if ok else "[FAIL] a safety or green-limit rule was violated")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())