"""Background jobs for long dashboard tasks (controller comparison, emergency scenario).
Streamlit-free: a job runs in a daemon thread and is polled via snapshot().
Cancellation is cooperative: it takes effect at the next report() call (between runs)."""
from __future__ import annotations

import hashlib
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config.settings import SimulationConfig
from evaluation.compare import compare_controllers
from evaluation.emergency_report import ambulance_comparison, save_emergency_report
from evaluation.reports import save_report
from simulation.adaptive_controller import AdaptiveController
from simulation.emergency_controller import EmergencyController
from simulation.fixed_time_controller import FixedTimeController
from simulation.sumo_environment import SumoEnvironment
from utils.logging_utils import get_logger

log = get_logger(__name__)

EMERGENCY_KPIS = ["avg_waiting_time_s", "avg_queue_length", "throughput_veh_per_hour",
                  "avg_travel_time_s", "avg_stopped_vehicles", "total_delay_s"]


class JobState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class JobCancelled(Exception):
    """Raised inside a job when the user pressed Cancel."""


class BackgroundJob:
    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = JobState.IDLE
        self._done = 0
        self._total = 0
        self._message = ""
        self._error: Optional[str] = None
        self._result: Any = None
        self._started: Optional[float] = None
        self._finished: Optional[float] = None

    def start(self, target: Callable[..., Any], *args, **kwargs) -> bool:
        with self._lock:
            if self._state == JobState.RUNNING:
                return False
            self._cancel.clear()
            self._state, self._done, self._total = JobState.RUNNING, 0, 0
            self._message, self._error, self._result = "Starting ...", None, None
            self._started, self._finished = time.time(), None
        self._thread = threading.Thread(target=self._run, args=(target, args, kwargs),
                                        name=f"job-{self.name}", daemon=True)
        self._thread.start()
        return True

    def _run(self, target, args, kwargs) -> None:
        try:
            result = target(self, *args, **kwargs)
        except JobCancelled:
            self._finish(JobState.CANCELLED, None, None)
        except Exception as exc:                      # shown to the user, never swallowed silently
            log.exception("Job '%s' failed", self.name)
            self._finish(JobState.ERROR, None, str(exc))
        else:
            self._finish(JobState.DONE, result, None)

    def _finish(self, state: JobState, result: Any, error: Optional[str]) -> None:
        with self._lock:
            self._state, self._result, self._error = state, result, error
            self._finished = time.time()

    def report(self, done: int, total: int, message: str = "") -> None:
        with self._lock:
            self._done, self._total, self._message = int(done), int(total), message
        if self._cancel.is_set():
            raise JobCancelled()

    def cancel(self) -> None:
        self._cancel.set()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            end = self._finished or time.time()
            return {"state": self._state.value, "done": self._done, "total": self._total,
                    "message": self._message, "error": self._error, "result": self._result,
                    "elapsed_s": (end - self._started) if self._started else 0.0}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_events(text: str) -> List[Tuple[float, str, str]]:
    """'100:N:straight, 260:E:left' -> [(100.0,'N','straight'), (260.0,'E','left')]"""
    out: List[Tuple[float, str, str]] = []
    for item in [t for t in text.replace("\n", ",").split(",") if t.strip()]:
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad event '{item.strip()}': expected time:approach:movement, e.g. 100:N:straight")
        try:
            t = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"Bad time in event '{item.strip()}'") from exc
        out.append((t, parts[1].strip().upper(), parts[2].strip().lower()))
    return out


def format_events(events: Sequence[Tuple[float, str, str]]) -> str:
    return ",".join(f"{t:g}:{a}:{m}" for t, a, m in events)


# --------------------------------------------------------------------------- #
# Job targets
# --------------------------------------------------------------------------- #
def comparison_target(job: BackgroundJob, cfg: SimulationConfig, checkpoint: Optional[Path],
                      seeds: Sequence[int], greens: Sequence[float]) -> dict:
    result = compare_controllers(cfg, checkpoint, seeds, greens, progress_cb=job.report)
    paths = save_report(result)
    return {"dir": str(paths["dir"]), "warnings": result.warnings}


def _emergency_run(cfg: SimulationConfig, base, enabled: bool, label: str, detection_range: float,
                   buffer_s: float, max_hold: float) -> dict:
    ctrl = EmergencyController(base, enabled=enabled, detection_range_m=detection_range,
                               clearance_buffer_s=buffer_s, max_hold_s=max_hold)
    with SumoEnvironment(cfg, label=label) as env:
        env.reset(seed=cfg.seed)
        route_hash = hashlib.sha256(Path(env.route_summary.path).read_bytes()).hexdigest()[:16]
        n_amb = len(env.route_summary.ambulances)
        summary = env.run(ctrl, on_step=ctrl.observe)
    return {"summary": summary, "events": ctrl.events_dataframe(), "timeline": ctrl.timeline_dataframe(),
            "stats": ctrl.stats(), "hash": route_hash, "ambulances_scheduled": n_amb,
            "messages": list(ctrl.messages)}


def emergency_target(job: BackgroundJob, cfg: SimulationConfig, base_kind: str,
                     checkpoint: Optional[Path], greens: Sequence[float], detection_range: float,
                     buffer_s: float, max_hold: float) -> dict:
    if base_kind == "dqn":
        base = AdaptiveController.from_checkpoint(checkpoint)
    else:
        base = FixedTimeController(tuple(greens))
    job.report(0, 2, "Run 1/2: WITHOUT priority")
    off = _emergency_run(cfg, base, False, "dash_emerg_off", detection_range, buffer_s, max_hold)
    job.report(1, 2, "Run 2/2: WITH priority")
    on = _emergency_run(cfg, base, True, "dash_emerg_on", detection_range, buffer_s, max_hold)
    job.report(2, 2, "finished")
    if off["hash"] != on["hash"]:
        raise RuntimeError("Demand differs between the two runs (route hashes differ); comparison invalid.")
    table = ambulance_comparison(off["events"], on["events"])
    kpi = pd.DataFrame({"no_priority": [off["summary"][k] for k in EMERGENCY_KPIS],
                        "priority": [on["summary"][k] for k in EMERGENCY_KPIS]}, index=EMERGENCY_KPIS)
    kpi["difference"] = kpi["priority"] - kpi["no_priority"]
    report = {"generated_by": "dashboard", "base": base_kind, "config": cfg.to_dict(),
              "events_scheduled": cfg.emergency_events,
              "detection_source": "simulation_ground_truth (simulated)", "route_hash": on["hash"],
              "ambulances_scheduled": on["ambulances_scheduled"],
              "stats_priority": on["stats"], "messages": on["messages"],
              "settings": {"detection_range_m": detection_range, "clearance_buffer_s": buffer_s,
                           "max_hold_s": max_hold}}
    out = save_emergency_report(report, {
        "events_no_priority": off["events"], "events_priority": on["events"],
        "timeline_no_priority": off["timeline"], "timeline_priority": on["timeline"],
        "ambulance_comparison": table, "kpi_comparison": kpi.reset_index(names="metric")})
    return {"dir": str(out)}


# --------------------------------------------------------------------------- #
_JOBS: Dict[str, BackgroundJob] = {}
_JOBS_LOCK = threading.Lock()


def _job(name: str) -> BackgroundJob:
    with _JOBS_LOCK:
        if name not in _JOBS:
            _JOBS[name] = BackgroundJob(name)
        return _JOBS[name]


def get_comparison_job() -> BackgroundJob:
    return _job("comparison")


def get_emergency_job() -> BackgroundJob:
    return _job("emergency")