"""Fixed-time vs DQN comparison under IDENTICAL demand, duration and seed.

Identical demand is VERIFIED: the generated route file is hashed per run and the two
controllers' hashes must match for each seed, otherwise ComparisonError is raised.
All numbers come from SUMO via TraCI. Nothing is estimated or filled in.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config.settings import SimulationConfig
from simulation.adaptive_controller import AdaptiveController
from simulation.fixed_time_controller import FixedTimeController
from simulation.sumo_environment import SumoEnvironment
from utils.logging_utils import get_logger

log = get_logger(__name__)

METRICS = ["avg_waiting_time_s", "avg_queue_length", "throughput_veh_per_hour",
           "avg_travel_time_s", "avg_stopped_vehicles", "total_delay_s"]
METRIC_LABELS = {
    "avg_waiting_time_s": "Average waiting time (s)",
    "avg_queue_length": "Average queue length (veh)",
    "throughput_veh_per_hour": "Throughput (veh/h)",
    "avg_travel_time_s": "Average travel time (s)",
    "avg_stopped_vehicles": "Stopped vehicles (avg)",
    "total_delay_s": "Total delay (s)",
}
HIGHER_IS_BETTER = {"throughput_veh_per_hour"}
CONTEXT = ["vehicles_entered", "vehicles_arrived", "vehicles_in_network_end", "phase_switches",
           "max_queue_length"]
CONTROLLERS = ("fixed_time", "dqn")


class ComparisonError(RuntimeError):
    """Raised when a comparison cannot be run fairly."""


def pct_improvement(metric: str, fixed: float, dqn: float) -> float:
    """Positive = DQN better. Higher-is-better metrics: (dqn-fixed)/|fixed|; others (fixed-dqn)/|fixed|."""
    if fixed is None or dqn is None or abs(float(fixed)) < 1e-9:
        return float("nan")
    gain = (dqn - fixed) if metric in HIGHER_IS_BETTER else (fixed - dqn)
    return 100.0 * float(gain) / abs(float(fixed))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run_one(controller, cfg: SimulationConfig, seed: int, label: str,
            on_step: Optional[Callable] = None) -> Dict:
    """One full simulation with one controller. Always closes SUMO."""
    with SumoEnvironment(cfg, label=label) as env:
        env.reset(seed=seed)
        rs = env.route_summary
        demand_hash = _file_hash(rs.path)
        summary = env.run(controller, on_step=on_step)
        ts = env.metrics.to_dataframe()
    extras = controller.stats() if hasattr(controller, "stats") else {}
    return {"summary": summary, "timeseries": ts, "demand_hash": demand_hash,
            "vehicles_generated": rs.total_vehicles, "per_approach": rs.per_approach,
            "extras": extras}


@dataclass
class ComparisonResult:
    config: dict
    seeds: List[int]
    fixed_greens: List[float]
    checkpoint: dict
    summaries: pd.DataFrame
    timeseries: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    dqn_extras: Dict[int, dict] = field(default_factory=dict)

    # ---- tables ----
    def per_seed_improvement(self) -> pd.DataFrame:
        rows = []
        for sd, g in self.summaries.groupby("seed"):
            f = g[g.controller == "fixed_time"].iloc[0]
            d = g[g.controller == "dqn"].iloc[0]
            row = {"seed": int(sd)}
            for m in METRICS:
                row[m] = pct_improvement(m, float(f[m]), float(d[m]))
            rows.append(row)
        return pd.DataFrame(rows).set_index("seed")

    def improvement_table(self) -> pd.DataFrame:
        mean = self.summaries.groupby("controller")[METRICS].mean()
        std = self.summaries.groupby("controller")[METRICS].std(ddof=1)
        per = self.per_seed_improvement()
        n = len(per)
        rows = []
        for m in METRICS:
            f, d = float(mean.loc["fixed_time", m]), float(mean.loc["dqn", m])
            rows.append({"metric": METRIC_LABELS[m], "fixed_time": f, "dqn": d,
                         "fixed_time_std": float(std.loc["fixed_time", m]),
                         "dqn_std": float(std.loc["dqn", m]),
                         "improvement_%": pct_improvement(m, f, d),
                         "dqn_better_in_seeds": f"{int((per[m] > 0).sum())}/{n}"})
        return pd.DataFrame(rows).set_index("metric")

    def context_table(self) -> pd.DataFrame:
        return self.summaries.groupby("controller")[CONTEXT].mean().T

    def timeseries_mean(self, column: str) -> pd.DataFrame:
        """Mean over seeds of a per-step column; index = sim time, columns = controllers."""
        return self.timeseries.groupby(["controller", "time"])[column].mean().unstack("controller")

    def to_dict(self) -> dict:
        return {"config": self.config, "seeds": self.seeds, "fixed_greens": self.fixed_greens,
                "checkpoint": self.checkpoint, "warnings": self.warnings,
                "improvement": self.improvement_table().reset_index().to_dict("records")}


def compare_controllers(cfg: SimulationConfig, checkpoint: Optional[Path] = None,
                        seeds: Sequence[int] = (20000,),
                        fixed_greens: Sequence[float] = (35, 10, 35, 10),
                        progress_cb: Optional[Callable[[int, int, str], None]] = None
                        ) -> ComparisonResult:
    cfg.validate()
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be a non-empty list of unique non-negative integers")
    fixed = FixedTimeController(fixed_greens)                       # validates green times
    dqn = AdaptiveController.from_checkpoint(checkpoint)            # raises CheckpointError
    warnings = dqn.compatibility_warnings(cfg)
    for w in warnings:
        log.warning("Compatibility: %s", w)

    total, done = 2 * len(seeds), 0
    rows: List[dict] = []
    ts_frames: List[pd.DataFrame] = []
    dqn_extras: Dict[int, dict] = {}
    for sd in seeds:
        hashes = {}
        for name, ctrl in (("fixed_time", fixed), ("dqn", dqn)):
            if progress_cb:
                progress_cb(done, total, f"{name} | seed {sd}")
            res = run_one(ctrl, cfg, sd, label=f"cmp_{name}")
            hashes[name] = res["demand_hash"]
            rows.append({"controller": name, "seed": sd, **res["summary"],
                         "vehicles_generated": res["vehicles_generated"],
                         "demand_hash": res["demand_hash"]})
            ts = res["timeseries"].copy()
            ts.insert(0, "seed", sd)
            ts.insert(0, "controller", name)
            ts_frames.append(ts)
            if name == "dqn":
                dqn_extras[sd] = res["extras"]
            done += 1
        if hashes["fixed_time"] != hashes["dqn"]:
            raise ComparisonError(f"Demand mismatch for seed {sd} (route hashes differ); "
                                  "comparison would be unfair.")
    if progress_cb:
        progress_cb(done, total, "finished")

    info = dqn.info()
    return ComparisonResult(
        config=cfg.to_dict(), seeds=seeds, fixed_greens=[float(g) for g in fixed.green_times],
        checkpoint=info, summaries=pd.DataFrame(rows), timeseries=pd.concat(ts_frames, ignore_index=True),
        warnings=warnings, dqn_extras=dqn_extras)