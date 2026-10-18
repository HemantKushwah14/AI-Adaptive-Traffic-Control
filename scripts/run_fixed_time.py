"""Phase 2 milestone: run the fixed-time controller on the real SUMO intersection.

Examples:
    python scripts\\run_fixed_time.py
    python scripts\\run_fixed_time.py --duration 900 --demand 700 --seed 7
    python scripts\\run_fixed_time.py --gui          (opens sumo-gui; press Play if paused)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import OUTPUTS_DIR, SimulationConfig, SumoNotFoundError  # noqa: E402
from simulation.fixed_time_controller import FixedTimeController  # noqa: E402
from simulation.sumo_environment import SumoEnvironment  # noqa: E402
from simulation.traci_controller import SimulationError  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=1800)
    p.add_argument("--demand", type=float, default=500.0, help="veh/h per approach")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()

    try:
        cfg = SimulationConfig.with_uniform_demand(args.demand, duration=args.duration,
                                                   seed=args.seed, gui=args.gui)
        with SumoEnvironment(cfg, label="fixed") as env:
            env.reset()
            rs = env.route_summary
            print(f"Generated {rs.total_vehicles} vehicles: {rs.per_approach}")
            summary = env.run(FixedTimeController())
            df = env.metrics.to_dataframe()
    except (SumoNotFoundError, SimulationError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    out_dir = OUTPUTS_DIR / "fixed_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"timeseries_seed{args.seed}.csv", index=False)
    (out_dir / f"summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Fixed-time controller summary (measured from SUMO via TraCI) ===")
    for k, v in summary.items():
        print(f"{k:28s} {v:.2f}" if isinstance(v, float) else f"{k:28s} {v}")
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())