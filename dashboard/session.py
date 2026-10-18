"""Process-wide simulation session. SUMO runs in a background thread so Streamlit reruns
never restart, block or duplicate it. All TraCI access is serialised by one lock.

Streamlit-free: test headless with  python scripts\\session_smoke_test.py
"""
from __future__ import annotations

import atexit
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import pandas as pd

from config.constants import APPROACHES, MOVEMENTS, PHASE_NAMES
from dashboard.sim_settings import build_sim_config, make_controller
from simulation.adaptive_controller import AdaptiveController
from simulation.fixed_time_controller import FixedTimeController
from simulation.sumo_environment import SumoEnvironment
from simulation.traci_controller import SignalMode, SimulationError
from utils.logging_utils import get_logger

log = get_logger(__name__)


class SessionState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class SimulationSession:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._run_evt = threading.Event()
        self.state = SessionState.IDLE
        self.error: Optional[str] = None
        self.warnings: List[str] = []
        self.speed = 5.0
        self._clear()
        atexit.register(self._shutdown)

    # ------------------------------------------------------------------ helpers
    def _clear(self) -> None:
        self.env: Optional[SumoEnvironment] = None
        self.ctrl = None
        self.settings: Optional[dict] = None
        self.cfg = None
        self.final_summary: Optional[dict] = None
        self.dispatched: List[dict] = []
        self._snapshot: Optional[dict] = None
        self._series: Deque[dict] = deque(maxlen=7200)
        self._dqn_hist: Deque[dict] = deque(maxlen=600)
        self._seen_decision = None
        self._q_sum = 0.0
        self._q_n = 0
        self._last_snap_wall = 0.0
        self._rate_wall = time.perf_counter()
        self._rate_sim = 0.0
        self._sim_rate = 0.0
        self._amb_counter = 0

    def _shutdown(self) -> None:
        try:
            self.reset()
        except Exception:  # interpreter exit
            pass

    def _fail(self, exc: Exception) -> str:
        log.error("Session error: %s", exc)
        self.state = SessionState.ERROR
        self.error = str(exc)
        return self.error

    @property
    def is_active(self) -> bool:
        return self.state in (SessionState.RUNNING, SessionState.PAUSED)

    # ------------------------------------------------------------------ lifecycle
    def start(self, settings: Dict[str, Any]) -> Optional[str]:
        """Start a fresh run. Returns None on success, else a readable error message."""
        self.reset()
        try:
            cfg = build_sim_config(settings)
            base, ctrl, warns = make_controller(settings, cfg)
            env = SumoEnvironment(cfg, label="dashboard")
        except Exception as exc:
            return self._fail(exc)
        try:
            env.reset()
            ctrl.reset()
        except Exception as exc:
            env.close()
            return self._fail(exc)
        with self._lock:
            self.env, self.ctrl, self.cfg = env, ctrl, cfg
            self.settings, self.warnings = dict(settings), warns
            self.speed = float(settings.get("speed", 5.0))
            self.error = None
            self.state = SessionState.RUNNING
            self._rate_wall, self._rate_sim = time.perf_counter(), 0.0
            self._publish(env, ctrl)
        self._stop_evt, self._run_evt = threading.Event(), threading.Event()
        self._run_evt.set()
        self._thread = threading.Thread(target=self._worker, name="sumo-session", daemon=True,
                                        args=(env, ctrl, self._stop_evt, self._run_evt))
        self._thread.start()
        return None

    def pause(self) -> None:
        with self._lock:
            if self.state == SessionState.RUNNING:
                self._run_evt.clear()
                self.state = SessionState.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self.state == SessionState.PAUSED:
                self.state = SessionState.RUNNING
                self._run_evt.set()

    def _stop_workers(self) -> None:
        self._stop_evt.set()
        self._run_evt.set()
        th = self._thread
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=15)
        self._thread = None

    def stop(self) -> None:
        """End the run but keep the collected data for display."""
        if not self.is_active:
            return
        self._stop_workers()
        with self._lock:
            if self.env is not None and self.is_active:
                self._finalise(self.env, SessionState.STOPPED)

    def reset(self) -> None:
        """Stop everything, close SUMO and forget all data."""
        self._stop_workers()
        with self._lock:
            if self.env is not None:
                self.env.close()
            self._clear()
            self.state = SessionState.IDLE
            self.error = None
            self.warnings = []

    def set_speed(self, speed: float) -> None:
        self.speed = max(float(speed), 0.0)

    # ------------------------------------------------------------------ worker
    @staticmethod
    def _sleep_until(deadline: float, stop_evt: threading.Event) -> None:
        while not stop_evt.is_set():
            d = deadline - time.perf_counter()
            if d <= 0:
                return
            time.sleep(min(d, 0.05))

    def _worker(self, env: SumoEnvironment, ctrl, stop_evt, run_evt) -> None:
        next_wall = time.perf_counter()
        try:
            while not stop_evt.is_set():
                if not run_evt.wait(0.1):                 # paused
                    next_wall = time.perf_counter()
                    continue
                with self._lock:
                    if stop_evt.is_set():
                        break
                    if env.done:
                        self._finalise(env, SessionState.FINISHED)
                        return
                    self._advance(env, ctrl)
                    if time.perf_counter() - self._last_snap_wall >= 0.08:
                        self._publish(env, ctrl)
                speed = self.speed
                if speed > 0:
                    next_wall += env.cfg.step_length / speed
                    self._sleep_until(next_wall, stop_evt)
                    if time.perf_counter() - next_wall > 0.5:      # fell behind: do not burst
                        next_wall = time.perf_counter()
                else:
                    time.sleep(0.0005)                             # "Max": just yield
        except Exception as exc:
            log.exception("Simulation worker crashed")
            with self._lock:
                self.error = f"Simulation stopped unexpectedly: {exc}"
                self.state = SessionState.ERROR
                try:
                    env.close()
                except Exception:
                    pass

    def _advance(self, env: SumoEnvironment, ctrl) -> None:
        """One simulation step: controller decision (only while GREEN), TraCI step, bookkeeping."""
        if env.signal.mode == SignalMode.GREEN:
            phase = ctrl.select_phase(env)
            if phase != env.signal.active_phase:
                env.signal.request_phase(phase)
        env.advance(env.cfg.step_length)
        ctrl.observe(env)
        rec = env.metrics.records[-1]
        self._q_sum += rec["queue_total"]
        self._q_n += 1
        row = {"time": rec["time"], "vehicles": rec["vehicles_in_network"],
               "stopped": rec["stopped_vehicles"], "queue": rec["queue_total"],
               "avg_wait_now": rec["avg_waiting_now"], "arrived": rec["arrived_step"],
               "cum_arrived": rec["cum_arrived"], "phase": rec["active_phase"], "mode": rec["mode"],
               "emergency_mode": ctrl.mode.value}
        for a in APPROACHES:
            row[f"queue_{a}"] = rec[f"queue_{a}"]
        self._series.append(row)
        last = getattr(getattr(ctrl, "base", None), "last", None)
        if last is not None and last is not self._seen_decision:
            self._seen_decision = last
            self._dqn_hist.append({"time": last.time, "q": list(last.q_values), "applied": last.applied,
                                   "requested": last.requested, "override": last.override_reason,
                                   "active_before": last.active_before})

    def _finalise(self, env: SumoEnvironment, state: SessionState) -> None:
        try:
            self.final_summary = env.summary()
        except Exception as exc:
            log.warning("Could not compute final summary: %s", exc)
        self.state = state
        self._publish(env, self.ctrl)
        env.close()

    # ------------------------------------------------------------------ snapshot
    def _recent_throughput(self, t: float) -> Optional[float]:
        if t < 30 or not self._series:
            return None
        cutoff, n = t - 60.0, 0
        for r in reversed(self._series):
            if r["time"] < cutoff:
                break
            n += r["arrived"]
        return n * 3600.0 / min(60.0, t)

    @staticmethod
    def _countdown(env, ctrl, si: dict) -> dict:
        em_mode = ctrl.mode.value
        if si["mode"] != "GREEN":
            return {"label": "Clearance: " + si["mode"].replace("_", "-").lower(),
                    "seconds": float(si["transition_remaining"]), "extra": ""}
        if em_mode != "NORMAL":
            return {"label": f"Emergency priority ({em_mode.lower()})", "seconds": None, "extra": ""}
        el, cfg, base = float(si["elapsed_green"]), env.cfg, ctrl.base
        if isinstance(base, FixedTimeController):
            return {"label": "Green remaining (fixed plan)",
                    "seconds": max(base.green_times[si["active_phase"]] - el, 0.0), "extra": ""}
        nxt = getattr(base, "_next_elapsed", None)
        secs = max((nxt if nxt is not None else cfg.decision_interval) - el, 0.0)
        lock = max(cfg.min_green - el, 0.0)
        return {"label": "Next DQN decision in", "seconds": secs,
                "extra": f"min-green lock {lock:.0f} s" if lock > 0 else "switching allowed"}

    def _publish(self, env: Optional[SumoEnvironment] = None, ctrl=None) -> None:
        env, ctrl = env or self.env, ctrl or self.ctrl
        if env is None or env.signal is None or env.observer is None:
            return
        obs, sig = env.observer, env.signal
        t = float(env.sim_time)
        veh = obs.vehicles
        vehicles = [{"id": v["id"], "x": v["x"], "y": v["y"], "angle": v["angle"], "type": v["type"],
                     "speed": v["speed"], "waiting": v["waiting"]} for v in veh.values()]
        fin = obs.finished
        waits = [f["waiting"] for f in fin] + [v["accum_waiting"] for v in veh.values()]
        lane = env.last_snapshot or {}
        si = sig.info()
        si["links"] = [tuple(x) for x in sig._signals]
        si["phase_names"] = list(PHASE_NAMES)
        approaches = {a: {"queue": lane["queue"][a], "count": lane["count"][a],
                          "waiting": lane["waiting"][a], "density": lane["density"][a],
                          "green": si["mode"] == "GREEN" and a in si["green_approaches"]}
                      for a in APPROACHES} if lane else {}
        base = ctrl.base
        dqn = None
        if isinstance(base, AdaptiveController):
            dqn = {"info": base.info(), "stats": base.stats(),
                   "last": base.last.to_dict() if base.last else None,
                   "history": list(self._dqn_hist)[-200:]}
        now = time.perf_counter()
        if now - self._rate_wall >= 1.0:
            self._sim_rate = (t - self._rate_sim) / (now - self._rate_wall)
            self._rate_wall, self._rate_sim = now, t
        cfg = env.cfg
        self._snapshot = {
            "time": t, "duration": cfg.duration, "progress": min(t / cfg.duration, 1.0),
            "controller": (self.settings or {}).get("controller", "?"),
            "priority_enabled": ctrl.enabled, "seed": cfg.seed,
            "vehicles": vehicles, "signal": si, "countdown": self._countdown(env, ctrl, si),
            "approaches": approaches, "lanes": lane.get("lanes", []),
            "kpi": {"vehicles_entered": len(fin) + len(veh), "vehicles_in_network": len(veh),
                    "vehicles_arrived": len(fin),
                    "avg_waiting_time_s": (sum(waits) / len(waits)) if waits else 0.0,
                    "avg_queue_length": (self._q_sum / self._q_n) if self._q_n else 0.0,
                    "queue_now": lane.get("queue_total", 0.0),
                    "stopped_now": sum(1 for v in veh.values() if v["speed"] < 0.1),
                    "throughput_now_vph": self._recent_throughput(t),
                    "throughput_avg_vph": len(fin) / (t / 3600.0) if t > 0 else 0.0,
                    "phase_switches": si["phase_switches"]},
            "emergency": ctrl.status(env), "dqn": dqn, "summary": self.final_summary,
            "dispatched": list(self.dispatched),
        }
        self._last_snap_wall = time.perf_counter()

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            if self._snapshot is None:
                return None
            out = dict(self._snapshot)
            out.update(state=self.state.value, error=self.error, speed=self.speed,
                       sim_rate=self._sim_rate)
            return out

    def series_df(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._series)
        return pd.DataFrame(rows)

    def emergency_events_df(self) -> pd.DataFrame:
        with self._lock:
            return self.ctrl.events_dataframe() if self.ctrl is not None else pd.DataFrame()

    def emergency_timeline_df(self) -> pd.DataFrame:
        with self._lock:
            return self.ctrl.timeline_dataframe() if self.ctrl is not None else pd.DataFrame()

    # ------------------------------------------------------------------ actions
    def dispatch_ambulance(self, approach: str, movement: str) -> str:
        """Inject a SIMULATED ambulance now (explicitly labelled demo vehicle)."""
        if approach not in APPROACHES or movement not in MOVEMENTS:
            raise ValueError("approach must be N/S/E/W and movement right/straight/left")
        with self._lock:
            if not self.is_active or self.env is None or not self.env.connection.is_open:
                raise SimulationError("Start the simulation before dispatching an ambulance.")
            self._amb_counter += 1
            vid = f"amb_dash_{self._amb_counter}"
            self.env.connection.conn.vehicle.add(vid, f"r_{approach}_{movement}", typeID="ambulance",
                                                 departLane="best", departSpeed="max")
            self.dispatched.append({"id": vid, "time": float(self.env.sim_time),
                                    "approach": approach, "movement": movement})
            self._publish(self.env, self.ctrl)
        return vid


_SESSION: Optional[SimulationSession] = None
_SESSION_LOCK = threading.Lock()


def get_session() -> SimulationSession:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = SimulationSession()
        return _SESSION