"""Training + testing efficiency report from YOUR training log and held-out SUMO runs.
    python scripts\\evaluate_efficiency.py
    python scripts\\evaluate_efficiency.py --seeds 5 --duration 600 --demand 500
Writes outputs\\efficiency_report.json and prints a verdict (measured, never assumed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.constants import PHASE_NAMES  # noqa: E402
from config.rl_config import TEST_SEED_BASE, TRAIN_LOG  # noqa: E402
from config.settings import OUTPUTS_DIR, SimulationConfig  # noqa: E402
from evaluation.compare import METRICS, compare_controllers  # noqa: E402
from simulation.adaptive_controller import default_checkpoint_path  # noqa: E402


def training_section() -> dict:
    out = {"available": False}
    if not TRAIN_LOG.exists():
        print("[WARN] no training log; train first.")
        return out
    df = pd.read_csv(TRAIN_LOG)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n = len(df)
    if n == 0:
        return out
    k = max(1, n // 4)
    first, last = df.head(k), df.tail(k)
    val = df.dropna(subset=["val_dqn_wait_s"])
    out = {
        "available": True, "episodes": int(n),
        "sec_per_episode_mean": float(df["wall_s"].mean()),
        "total_train_minutes": float(df["wall_s"].sum() / 60),
        "decisions_per_sec": float(df["decisions"].sum() / max(df["wall_s"].sum(), 1e-9)),
        "reward_first_quarter": float(first["total_reward"].mean()),
        "reward_last_quarter": float(last["total_reward"].mean()),
        "wait_first_quarter_s": float(first["avg_wait_s"].mean()),
        "wait_last_quarter_s": float(last["avg_wait_s"].mean()),
        "final_epsilon": float(df["epsilon"].iloc[-1]),
        "loss_finite": bool(np.isfinite(df["mean_loss"].dropna()).all()),
        "val_points": int(len(val)),
    }
    if len(val):
        out["val_dqn_wait_first_s"] = float(val["val_dqn_wait_s"].iloc[0])
        out["val_dqn_wait_last_s"] = float(val["val_dqn_wait_s"].iloc[-1])
        out["val_fixed_wait_s"] = float(val["val_fixed_wait_s"].dropna().iloc[0]) if val["val_fixed_wait_s"].notna().any() else None
    print("\n=== TRAINING (from your training_log.csv) ===")
    print(f"episodes: {out['episodes']} | {out['sec_per_episode_mean']:.1f} s/episode | total {out['total_train_minutes']:.1f} min | "
          f"{out['decisions_per_sec']:.1f} decisions/s")
    print(f"reward  first quarter {out['reward_first_quarter']:.1f} -> last quarter {out['reward_last_quarter']:.1f}")
    print(f"avg wait first quarter {out['wait_first_quarter_s']:.1f}s -> last quarter {out['wait_last_quarter_s']:.1f}s "
          "(demand varies per episode, so this is noisy)")
    if "val_dqn_wait_last_s" in out:
        print(f"validation wait (greedy, fixed seeds): {out['val_dqn_wait_first_s']:.1f}s -> {out['val_dqn_wait_last_s']:.1f}s "
              f"| fixed-time baseline {out['val_fixed_wait_s']}")
    print(f"final epsilon {out['final_epsilon']:.2f} | loss finite: {out['loss_finite']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--demand", type=float, default=500.0)
    a = ap.parse_args()

    train = training_section()
    ckpt = default_checkpoint_path()
    cfg = SimulationConfig.with_uniform_demand(a.demand, duration=a.duration)
    seeds = [TEST_SEED_BASE + i for i in range(a.seeds)]
    print(f"\n=== TESTING on held-out seeds {seeds} | checkpoint {ckpt.name} ===")
    t0 = time.time()
    res = compare_controllers(cfg, ckpt, seeds, progress_cb=lambda d, t, m: print(f"  [{d}/{t}] {m}"))
    wall = time.time() - t0
    sim_seconds = 2 * len(seeds) * a.duration
    imp = res.improvement_table()
    print(imp[["fixed_time", "dqn", "improvement_%", "dqn_better_in_seeds"]].round(2).to_string())
    print(f"\ntesting speed: {sim_seconds / wall:.0f} simulated s per real s ({wall:.0f} s total)")

    usage = {}
    for sd, ex in res.dqn_extras.items():
        for k, v in ex.get("phase_usage", {}).items():
            usage[k] = usage.get(k, 0) + v
    total = max(sum(usage.values()), 1)
    share = {p: usage.get(p, 0) / total for p in PHASE_NAMES}
    print("DQN phase usage:", {k: f"{v:.0%}" for k, v in share.items()})

    ts = res.timeseries
    dq = ts[ts.controller == "dqn"]
    worst_queue = {a_: float(dq[f"queue_{a_}"].mean()) for a_ in "NSEW"}
    print("DQN mean queue per approach:", {k: round(v, 1) for k, v in worst_queue.items()})

    wait = imp.loc["Average waiting time (s)"]
    stuck = imp.loc["Average travel time (s)"]
    ctx = res.context_table()
    checks = {
        "loss finite during training": train.get("loss_finite", False),
        "trained >= 30 episodes": train.get("episodes", 0) >= 30,
        "DQN uses all 4 phases (each >= 5%)": all(v >= 0.05 for v in share.values()),
        "no approach starved (max/min queue <= 4x)": max(worst_queue.values()) <= 4 * max(min(worst_queue.values()), 1.0),
        "DQN waiting time lower than fixed-time": wait["improvement_%"] > 0,
        "DQN better in a majority of seeds": int(wait["dqn_better_in_seeds"].split("/")[0]) > len(seeds) / 2,
        "DQN does not strand more vehicles than fixed-time":
            ctx.loc["vehicles_in_network_end", "dqn"] <= ctx.loc["vehicles_in_network_end", "fixed_time"] + 2,
    }
    print("\n=== VERDICT (each line is measured) ===")
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    passed = sum(checks.values())
    print(f"\n{passed}/{len(checks)} criteria met. Waiting-time change vs fixed-time: {wait['improvement_%']:+.1f}% "
          f"(positive = DQN better), better in {wait['dqn_better_in_seeds']} seeds.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "efficiency_report.json").write_text(json.dumps({
        "training": train, "test_seeds": seeds, "duration": a.duration, "demand": a.demand,
        "improvement": imp.reset_index().to_dict("records"), "phase_share": share,
        "queue_per_approach": worst_queue, "testing_sim_seconds_per_real_second": sim_seconds / wall,
        "criteria": {k: bool(v) for k, v in checks.items()}}, indent=2, default=str), encoding="utf-8")
    print("Saved outputs\\efficiency_report.json")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())