"""Greedy DQN evaluation on SUMO, optionally side by side with the fixed-time baseline.

    python -m reinforcement_learning.evaluate --seeds 3
    python -m reinforcement_learning.evaluate --checkpoint models\\dqn\\best.pt --duration 900 --demand 600

Both controllers see IDENTICAL demand (same SimulationConfig + seed => identical route file).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.constants import PHASE_NAMES
from config.rl_config import BEST_CKPT, DQN_OUT_DIR, LAST_CKPT, TEST_SEED_BASE
from config.settings import SimulationConfig, SumoNotFoundError
from reinforcement_learning.action_mask import legal_action_mask
from reinforcement_learning.dqn_agent import CheckpointError, DQNAgent
from simulation.fixed_time_controller import FixedTimeController
from simulation.sumo_environment import SumoEnvironment
from simulation.traci_controller import SimulationError

METRICS = ["avg_waiting_time_s", "avg_queue_length", "throughput_veh_per_hour",
           "avg_travel_time_s", "avg_stopped_vehicles", "total_delay_s"]
HIGHER_IS_BETTER = {"throughput_veh_per_hour"}


def run_dqn_episode(env: SumoEnvironment, agent: DQNAgent, seed: Optional[int] = None,
                    on_step=None) -> Tuple[Dict, Dict]:
    """One full greedy episode. Returns (summary, extras). Bounded loop."""
    state = env.reset(seed=seed)
    usage: Counter = Counter()
    overrides = decisions = 0
    limit = int(env.cfg.duration / env.cfg.decision_interval) + 200
    while not env.done:
        if decisions >= limit:
            raise SimulationError("Evaluation exceeded its step limit (possible stuck simulation).")
        action = agent.select_action(state, legal_action_mask(env), greedy=True)
        state, _done, info = env.step(action)
        decisions += 1
        overrides += int(info["overridden"])
        usage[PHASE_NAMES[info["action_applied"]]] += 1
        if on_step is not None:
            on_step(env, info)
    return env.summary(), {"decisions": decisions, "overrides": overrides, "phase_usage": dict(usage)}


def run_fixed_episode(env: SumoEnvironment, seed: Optional[int] = None) -> Dict:
    env.reset(seed=seed)
    return env.run(FixedTimeController())


def evaluate_checkpoint(checkpoint: Path, cfg: SimulationConfig, seeds: List[int],
                        compare_fixed: bool = True) -> pd.DataFrame:
    agent = DQNAgent.from_checkpoint(checkpoint, with_buffer=False)
    rows: List[dict] = []
    with SumoEnvironment(cfg, label="eval_dqn") as env_d:
        for sd in seeds:
            summ, extra = run_dqn_episode(env_d, agent, seed=sd)
            rows.append({"controller": "dqn", "seed": sd, **summ, "phase_usage": str(extra["phase_usage"]),
                         "overrides": extra["overrides"]})
            print(f"  dqn   seed {sd}: wait {summ['avg_waiting_time_s']:.1f}s  "
                  f"queue {summ['avg_queue_length']:.1f}  thr {summ['throughput_veh_per_hour']:.0f}/h")
    if compare_fixed:
        with SumoEnvironment(cfg, label="eval_fixed") as env_f:
            for sd in seeds:
                summ = run_fixed_episode(env_f, seed=sd)
                rows.append({"controller": "fixed_time", "seed": sd, **summ})
                print(f"  fixed seed {sd}: wait {summ['avg_waiting_time_s']:.1f}s  "
                      f"queue {summ['avg_queue_length']:.1f}  thr {summ['throughput_veh_per_hour']:.0f}/h")
    return pd.DataFrame(rows)


def improvement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per controller + percent improvement of DQN over fixed-time (positive = DQN better)."""
    mean = df.groupby("controller")[METRICS].mean().T
    if {"dqn", "fixed_time"} <= set(mean.columns):
        def pct(row):
            f, d = row["fixed_time"], row["dqn"]
            if abs(f) < 1e-9:
                return float("nan")
            gain = (d - f) if row.name in HIGHER_IS_BETTER else (f - d)
            return 100.0 * gain / abs(f)
        mean["improvement_%"] = mean.apply(pct, axis=1)
    return mean


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a DQN checkpoint on held-out SUMO seeds.")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--seeds", type=int, default=3, help="number of test seeds")
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--demand", type=float, default=500.0, help="veh/h per approach")
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--max-green", type=int, default=60)
    p.add_argument("--no-fixed", action="store_true")
    a = p.parse_args()

    ckpt = a.checkpoint or (BEST_CKPT if BEST_CKPT.exists() else LAST_CKPT)
    try:
        cfg = SimulationConfig.with_uniform_demand(a.demand, duration=a.duration,
                                                   min_green=a.min_green, max_green=a.max_green)
        cfg.validate()
        seeds = [TEST_SEED_BASE + i for i in range(max(1, a.seeds))]
        print(f"Checkpoint: {ckpt}\nTest seeds: {seeds} (never used for training/validation)")
        df = evaluate_checkpoint(ckpt, cfg, seeds, compare_fixed=not a.no_fixed)
    except (CheckpointError, SumoNotFoundError, SimulationError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    DQN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = DQN_OUT_DIR / f"eval_{Path(ckpt).stem}.csv"
    df.to_csv(out, index=False)
    print("\nMean over seeds (measured from SUMO):")
    print(improvement_table(df).round(2).to_string())
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())