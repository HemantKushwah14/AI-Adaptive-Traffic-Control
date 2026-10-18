"""Phase 5 milestone: DQN vs fixed-time on identical SUMO demand.

    python scripts\\run_comparison.py --seeds 3 --duration 900 --demand 500
    python scripts\\run_comparison.py --checkpoint models\\dqn\\last.pt --duration 600 --seeds 2
    python scripts\\run_comparison.py --demand-n 800 --demand-s 800 --demand-e 300 --demand-w 300
    python scripts\\run_comparison.py --fixed-greens 30,12,30,12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.constants import APPROACHES  # noqa: E402
from config.rl_config import TEST_SEED_BASE  # noqa: E402
from config.settings import SimulationConfig, SumoNotFoundError  # noqa: E402
from evaluation.compare import ComparisonError, compare_controllers  # noqa: E402
from evaluation.reports import save_report  # noqa: E402
from reinforcement_learning.dqn_agent import CheckpointError  # noqa: E402
from simulation.traci_controller import SimulationError  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Compare fixed-time and DQN control on identical demand.")
    p.add_argument("--checkpoint", type=Path, default=None, help="default: models/dqn/best.pt, else last.pt")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (1..20)")
    p.add_argument("--base-seed", type=int, default=TEST_SEED_BASE)
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--demand", type=float, default=500.0, help="veh/h per approach")
    for a in APPROACHES:
        p.add_argument(f"--demand-{a.lower()}", type=float, default=None, help=f"override {a} approach")
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--max-green", type=int, default=60)
    p.add_argument("--decision-interval", type=int, default=5)
    p.add_argument("--fixed-greens", default="35,10,35,10",
                   help="fixed-time greens for NS_THROUGH,NS_LEFT,EW_THROUGH,EW_LEFT (seconds)")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    if not 1 <= a.seeds <= 20:
        print("ERROR: --seeds must be within 1..20")
        return 1
    try:
        greens = tuple(float(x) for x in a.fixed_greens.split(","))
        cfg = SimulationConfig.with_uniform_demand(
            a.demand, duration=a.duration, min_green=a.min_green, max_green=a.max_green,
            decision_interval=a.decision_interval)
        for ap in APPROACHES:
            v = getattr(a, f"demand_{ap.lower()}")
            if v is not None:
                cfg.demand_vph[ap] = float(v)
        cfg.validate()
        seeds = [a.base_seed + i for i in range(a.seeds)]
        if min(seeds) < TEST_SEED_BASE:
            print(f"NOTE: seeds {seeds} include values below {TEST_SEED_BASE}; these may overlap "
                  "training/validation seeds, so results are not strictly held-out.")
        print(f"Seeds: {seeds} | duration {cfg.duration}s | demand {cfg.demand_vph}")

        def progress(done: int, total: int, msg: str) -> None:
            print(f"[{done}/{total}] {msg}")

        result = compare_controllers(cfg, a.checkpoint, seeds, greens, progress)
    except (CheckpointError, SumoNotFoundError, SimulationError, ComparisonError,
            FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\nCheckpoint: {result.checkpoint['checkpoint']} "
          f"(episodes trained: {result.checkpoint['episodes_trained']})")
    for w in result.warnings:
        print(f"WARNING: {w}")
    print("\n=== Mean over seeds (measured from SUMO; improvement > 0 means DQN is better) ===")
    print(result.improvement_table().round(2).to_string())
    print("\n=== Context (mean over seeds) ===")
    print(result.context_table().round(1).to_string())
    print("\n=== Per-seed improvement % ===")
    print(result.per_seed_improvement().round(1).to_string())
    if not a.no_save:
        paths = save_report(result)
        print(f"\nSaved report to {paths['dir']}\n  (also copied to outputs\\comparison\\latest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())