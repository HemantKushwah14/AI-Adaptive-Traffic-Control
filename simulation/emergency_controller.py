"""Emergency priority controller wrapping a base controller (FixedTime or DQN).

Interface == other controllers (reset / select_phase / stats) so env.run(controller, on_step=
controller.observe) works. The signal is still driven by SignalManager, so every change goes
green -> yellow -> all-red -> green; this controller only chooses WHICH phase to request.

Detection sources (explicitly labelled):
  SimulationTruthSource  - reads ambulance vehicle type via TraCI (SIMULATED detection)
  ExternalDetectionSource- accepts sightings from a real camera/vision pipeline

Logic self-test with a stub environment (NOT a traffic result):
    python -m simulation.emergency_controller --selftest
"""
from __future__ import annotations

import sys
import types
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd

from config.constants import APPROACH_NAMES, APPROACHES, LANE_INDEX, NUM_LANES, PHASE_GREEN, PHASE_NAMES
from config.emergency_config import (DEFAULT_CLEARANCE_BUFFER_S, DEFAULT_DETECTION_RANGE_M,
                                     DEFAULT_MAX_HOLD_S)
from utils.logging_utils import get_logger

log = get_logger(__name__)

EVENT_COLUMNS = ["vehicle_id", "approach", "lane_index", "required_phase", "source", "priority_enabled",
                 "detected_time_s", "detected_distance_m", "granted_time_s", "wait_for_green_s",
                 "cleared_time_s", "detection_to_clear_s", "accum_waiting_s", "status", "timed_out",
                 "progress"]


class EmergencyMode(str, Enum):
    NORMAL = "NORMAL"
    PREEMPTING = "PREEMPTING"
    EMERGENCY_GREEN = "EMERGENCY_GREEN"
    RECOVERY = "RECOVERY"


def required_phase(approach: str, lane_index: int) -> int:
    """The legal phase that gives green to this approach/lane."""
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach '{approach}'")
    for p, g in PHASE_GREEN.items():
        if lane_index in g.get(approach, ()):
            return p
    raise ValueError(f"No phase serves approach {approach}, lane {lane_index}")


# --------------------------------------------------------------------------- #
# Detection sources
# --------------------------------------------------------------------------- #
@dataclass
class EmergencyRequest:
    vehicle_id: str
    approach: str
    lane_index: int
    distance_to_stop: Optional[float]
    speed: float
    source: str
    confidence: Optional[float] = None
    accum_waiting: Optional[float] = None


class DetectionSource(Protocol):
    name: str

    def reset(self) -> None: ...

    def poll(self, env) -> List[EmergencyRequest]: ...


class SimulationTruthSource:
    """SIMULATED detection: ambulances seen through TraCI within `detection_range_m`."""
    name = "simulation_ground_truth"

    def __init__(self, detection_range_m: float = DEFAULT_DETECTION_RANGE_M) -> None:
        if detection_range_m <= 0:
            raise ValueError("detection_range_m must be > 0")
        self.detection_range_m = float(detection_range_m)
        self.reset()

    def reset(self) -> None:
        self._key = None
        self._movements: Dict[str, str] = {}

    def _movement_map(self, env) -> Dict[str, str]:
        rs = getattr(env, "route_summary", None)
        if rs is not self._key:
            self._key = rs
            self._movements = {a["id"]: a["movement"] for a in (rs.ambulances if rs else [])}
        return self._movements

    def poll(self, env) -> List[EmergencyRequest]:
        movements, out = self._movement_map(env), []
        for v in env.observer.emergency_vehicles():
            dist = v.get("dist_to_stop")
            if v.get("approach") is None or dist is None or dist > self.detection_range_m:
                continue
            if v["id"] in movements:
                lane = LANE_INDEX[movements[v["id"]]]
            else:
                try:
                    lane = int(str(v["lane"]).rsplit("_", 1)[1])
                except (ValueError, IndexError):
                    lane = LANE_INDEX["straight"]
            out.append(EmergencyRequest(v["id"], v["approach"], lane, float(dist), float(v["speed"]),
                                        self.name, None, float(v.get("accum_waiting", 0.0))))
        return out


class ExternalDetectionSource:
    """Receives sightings from a real vision pipeline. A request stays active while it is
    refreshed at least every `ttl_s` simulated seconds, then counts as cleared."""
    name = "vision_detection"

    def __init__(self, ttl_s: float = 3.0) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        self.ttl_s = float(ttl_s)
        self.reset()

    def reset(self) -> None:
        self._items: Dict[Tuple[str, int], dict] = {}
        self._counter = 0

    def submit(self, approach: str, now: float, lane_index: Optional[int] = None,
               confidence: Optional[float] = None) -> None:
        if approach not in APPROACHES:
            raise ValueError(f"Unknown approach '{approach}'")
        lane = LANE_INDEX["straight"] if lane_index is None else int(lane_index)
        if not 0 <= lane < NUM_LANES:
            raise ValueError(f"lane_index must be within 0..{NUM_LANES - 1}")
        key = (approach, lane)
        if key in self._items:
            self._items[key].update(last=float(now), conf=confidence)
        else:
            self._counter += 1
            self._items[key] = {"last": float(now), "conf": confidence, "n": self._counter}

    def poll(self, env) -> List[EmergencyRequest]:
        now, out = float(env.sim_time), []
        for key, it in list(self._items.items()):
            if now - it["last"] > self.ttl_s:
                del self._items[key]
                continue
            out.append(EmergencyRequest(f"vision_{key[0]}_{key[1]}_{it['n']}", key[0], key[1], None, 0.0,
                                        self.name, it["conf"]))
        return out


def submit_vision_sightings(source: ExternalDetectionSource, now: float, sightings) -> int:
    """Forward confirmed AmbulanceSighting objects (computer_vision.ambulance_detector)."""
    n = 0
    for s in sightings:
        if s.approach is None:
            continue
        source.submit(s.approach, now, s.lane_index, s.confidence)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass
class EmergencyEvent:
    vehicle_id: str
    approach: str
    lane_index: int
    phase: int
    source: str
    detected_time: float
    detected_distance: Optional[float]
    priority_enabled: bool
    status: str = "waiting"            # waiting | green | cleared | timeout
    granted_time: Optional[float] = None
    cleared_time: Optional[float] = None
    last_distance: Optional[float] = None
    last_accum_waiting: Optional[float] = None
    confidence: Optional[float] = None
    timed_out: bool = False

    def progress(self) -> float:
        if self.status == "cleared" or self.cleared_time is not None:
            return 1.0
        if self.detected_distance and self.last_distance is not None and self.detected_distance > 0:
            return float(min(max(1.0 - self.last_distance / self.detected_distance, 0.0), 1.0))
        return 0.0

    def to_dict(self) -> dict:
        return {
            "vehicle_id": self.vehicle_id, "approach": self.approach, "lane_index": self.lane_index,
            "required_phase": PHASE_NAMES[self.phase], "source": self.source,
            "priority_enabled": self.priority_enabled, "detected_time_s": self.detected_time,
            "detected_distance_m": self.detected_distance, "granted_time_s": self.granted_time,
            "wait_for_green_s": None if self.granted_time is None else self.granted_time - self.detected_time,
            "cleared_time_s": self.cleared_time,
            "detection_to_clear_s": None if self.cleared_time is None else self.cleared_time - self.detected_time,
            "accum_waiting_s": self.last_accum_waiting, "status": self.status,
            "timed_out": self.timed_out, "progress": self.progress()}


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class EmergencyController:
    name = "emergency_priority"

    def __init__(self, base, sources: Optional[Sequence[DetectionSource]] = None, enabled: bool = True,
                 detection_range_m: float = DEFAULT_DETECTION_RANGE_M, preempt_min_green: float = 0.0,
                 clearance_buffer_s: float = DEFAULT_CLEARANCE_BUFFER_S,
                 max_hold_s: float = DEFAULT_MAX_HOLD_S) -> None:
        if preempt_min_green < 0 or clearance_buffer_s < 0:
            raise ValueError("preempt_min_green and clearance_buffer_s must be >= 0")
        if max_hold_s < 5:
            raise ValueError("max_hold_s must be >= 5 seconds")
        self.base = base
        self.sources: List[DetectionSource] = list(sources) if sources else [SimulationTruthSource(detection_range_m)]
        self.enabled = bool(enabled)
        self.preempt_min_green = float(preempt_min_green)
        self.clearance_buffer_s = float(clearance_buffer_s)
        self.max_hold_s = float(max_hold_s)
        self.timeline: Deque[dict] = deque(maxlen=50_000)
        self.messages: Deque[dict] = deque(maxlen=300)
        self.reset()

    # ---------------- lifecycle ----------------
    def reset(self) -> None:
        self.base.reset()
        for s in self.sources:
            s.reset()
        self.mode = EmergencyMode.NORMAL
        self.events: Dict[str, EmergencyEvent] = {}
        self.messages.clear()
        self.timeline.clear()
        self.activations = 0
        self.timeouts = 0
        self.time_in_mode: Counter = Counter()
        self._last_obs: Optional[float] = None
        self._green_since: Optional[float] = None
        self._recovery_until: Optional[float] = None
        self._serving_phase: Optional[int] = None
        self._target_phase: Optional[int] = None
        self._requested_phase: Optional[int] = None

    def _note(self, t: float, msg: str) -> None:
        self.messages.append({"time": float(t), "message": msg})
        log.info("[t=%.0f] %s", t, msg)

    # ---------------- observation (call once per simulation step) ----------------
    def observe(self, env) -> None:
        t = float(env.sim_time)
        if self._last_obs == t:
            return
        self._last_obs = t
        present: Dict[str, EmergencyRequest] = {}
        for src in self.sources:
            for r in src.poll(env):
                present[r.vehicle_id] = r
        for vid, r in present.items():
            ev = self.events.get(vid)
            if ev is None:
                ev = EmergencyEvent(vid, r.approach, r.lane_index, required_phase(r.approach, r.lane_index),
                                    r.source, t, r.distance_to_stop, self.enabled)
                self.events[vid] = ev
                self._note(t, f"Emergency vehicle {vid} detected on the {APPROACH_NAMES[r.approach]} approach "
                              f"[{r.source}]; needs {PHASE_NAMES[ev.phase]}.")
            ev.last_distance, ev.last_accum_waiting, ev.confidence = r.distance_to_stop, r.accum_waiting, r.confidence
        for ev in self.events.values():
            if ev.status in ("waiting", "green", "timeout") and ev.cleared_time is None \
                    and ev.vehicle_id not in present:
                ev.cleared_time = t
                if ev.status != "timeout":
                    ev.status = "cleared"
                self._note(t, f"{ev.vehicle_id} cleared the {APPROACH_NAMES[ev.approach]} approach "
                              f"({t - ev.detected_time:.0f} s after detection).")
        self.time_in_mode[self.mode.value] += float(env.cfg.step_length)
        sig = env.signal
        self.timeline.append({
            "time": t, "mode": self.mode.value, "active_phase": sig.active_phase,
            "signal_mode": sig.mode.value, "active_requests": sum(e.status in ("waiting", "green") for e in self.events.values()),
            "target_phase": self._target_phase if self._target_phase is not None else -1})

    # ---------------- control ----------------
    @staticmethod
    def _eta_key(e: EmergencyEvent):
        d = e.last_distance if e.last_distance is not None else 0.0
        return (d / 3.0 if d <= 0 else d / max(3.0, 0.0), e.detected_time)

    def select_phase(self, env) -> int:
        """Called only while the signal is GREEN (by SumoEnvironment.run)."""
        self.observe(env)
        if not self.enabled:
            return self.base.select_phase(env)
        sig, now = env.signal, float(env.sim_time)
        active = [e for e in self.events.values() if e.status in ("waiting", "green")]
        if not active:
            return self._no_active(env, now)
        self._recovery_until = None
        serving = [e for e in active if e.phase == sig.active_phase]
        if self.mode == EmergencyMode.EMERGENCY_GREEN and serving:
            target = sig.active_phase
        else:
            target = min(active, key=self._eta_key).phase
        self._target_phase = target
        if self.mode in (EmergencyMode.NORMAL, EmergencyMode.RECOVERY):
            self.mode = EmergencyMode.PREEMPTING
            self.activations += 1
            self._note(now, "Emergency priority activated.")
        if sig.active_phase == target:
            return self._hold(env, now, active, target)
        if sig.elapsed_green + 1e-9 < self.preempt_min_green:
            return sig.active_phase
        if self.mode == EmergencyMode.EMERGENCY_GREEN:       # finished one phase, next ambulance elsewhere
            self.mode = EmergencyMode.PREEMPTING
            self._serving_phase = self._green_since = None
        if self._requested_phase != target:
            self._requested_phase = target
            self._note(now, f"Requesting {PHASE_NAMES[target]}: yellow -> all-red -> green "
                            "(safe transition enforced by the signal manager).")
        return target

    def _hold(self, env, now: float, active: List[EmergencyEvent], phase: int) -> int:
        if self.mode != EmergencyMode.EMERGENCY_GREEN or self._serving_phase != phase:
            self.mode, self._serving_phase, self._green_since = EmergencyMode.EMERGENCY_GREEN, phase, now
            self._requested_phase = None
        for e in active:
            if e.phase == phase and e.status == "waiting":
                e.status, e.granted_time = "green", now
                self._note(now, f"Green granted to {e.vehicle_id} ({APPROACH_NAMES[e.approach]}) "
                                f"{now - e.detected_time:.0f} s after detection.")
        if now - (self._green_since or now) > self.max_hold_s:
            for e in active:
                if e.phase == phase:
                    e.status, e.timed_out = "timeout", True
                    self.timeouts += 1
            self._note(now, f"Emergency hold exceeded {self.max_hold_s:.0f} s; forced release to protect "
                            "other approaches.")
            self._enter_recovery(now)
        return phase if self.mode == EmergencyMode.EMERGENCY_GREEN else env.signal.active_phase

    def _enter_recovery(self, now: float) -> None:
        self.mode = EmergencyMode.RECOVERY
        self._recovery_until = now + self.clearance_buffer_s
        self._serving_phase = self._green_since = None

    def _no_active(self, env, now: float) -> int:
        sig = env.signal
        if self.mode == EmergencyMode.EMERGENCY_GREEN:
            self._enter_recovery(now)
            self._note(now, f"Emergency cleared; holding green {self.clearance_buffer_s:.0f} s, then "
                            f"restoring {getattr(self.base, 'name', 'base')} control.")
        elif self.mode == EmergencyMode.PREEMPTING:
            self.mode = EmergencyMode.NORMAL
            self._note(now, "Emergency request withdrawn before green was granted.")
        self._requested_phase = None
        self._target_phase = None
        if self.mode == EmergencyMode.RECOVERY:
            if now < (self._recovery_until if self._recovery_until is not None else now) - 1e-9:
                return sig.active_phase
            self.mode = EmergencyMode.NORMAL
            self._note(now, f"Normal control restored ({getattr(self.base, 'name', 'base')}).")
        return self.base.select_phase(env)

    # ---------------- information ----------------
    def status(self, env=None) -> Dict:
        active = [e for e in self.events.values() if e.status in ("waiting", "green")]
        sig = env.signal.info() if env is not None and getattr(env, "signal", None) is not None else {}
        now = float(env.sim_time) if env is not None else 0.0
        tgt = self._target_phase
        progress = [e.progress() for e in active]
        return {
            "enabled": self.enabled, "mode": self.mode.value, "base_controller": getattr(self.base, "name", "?"),
            "target_phase": PHASE_NAMES[tgt] if tgt is not None else None,
            "target_approach": min(active, key=self._eta_key).approach if active else None,
            "active_requests": [e.to_dict() for e in active],
            "clearance_progress": float(sum(progress) / len(progress)) if progress else None,
            "hold_s": (now - self._green_since) if self._green_since is not None else 0.0,
            "max_hold_s": self.max_hold_s,
            "recovery_remaining_s": max((self._recovery_until or now) - now, 0.0) if self.mode == EmergencyMode.RECOVERY else 0.0,
            "signal_mode": sig.get("mode"),
            "signal_transition": {"YELLOW": "Yellow clearance", "ALL_RED": "All-red clearance",
                                  "GREEN": "Green"}.get(sig.get("mode"), "-"),
            "transition_remaining_s": sig.get("transition_remaining", 0.0),
            "activations": self.activations, "timeouts": self.timeouts,
            "time_in_mode_s": dict(self.time_in_mode), "log": list(self.messages)[-12:]}

    def stats(self) -> Dict:
        base_stats = self.base.stats() if hasattr(self.base, "stats") else {}
        return {"activations": self.activations, "timeouts": self.timeouts, "events": len(self.events),
                "time_in_mode_s": dict(self.time_in_mode), "base": base_stats}

    def events_dataframe(self) -> pd.DataFrame:
        rows = [e.to_dict() for e in self.events.values()]
        return pd.DataFrame(rows, columns=EVENT_COLUMNS)

    def timeline_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.timeline))


# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Stub signal + stub ambulance (NO SUMO). Checks that priority reaches green sooner than the
    plain fixed-time cycle, hold/clear/restore works, and requests go through request_phase only."""
    from config.settings import SimulationConfig
    from simulation.fixed_time_controller import FixedTimeController
    NS = types.SimpleNamespace

    class Sig:
        def __init__(self):
            self.active_phase = self.target_phase = 0
            self.elapsed_green, self.mode, self._t = 0.0, NS(value="GREEN"), 0.0

        def request_phase(self, p):
            if self.mode.value == "GREEN" and p != self.active_phase:
                self.target_phase, self.mode, self._t = p, NS(value="YELLOW"), 4.0

        def tick(self, dt):
            if self.mode.value == "GREEN":
                self.elapsed_green += dt
            else:
                self._t -= dt
                if self._t <= 1e-9:
                    self.active_phase, self.mode, self.elapsed_green = self.target_phase, NS(value="GREEN"), 0.0

        def info(self):
            return {"mode": self.mode.value, "transition_remaining": max(self._t, 0.0)}

    class Obs:
        def __init__(self, sig):
            self.sig, self.dist, self.wait, self.exit_time, self.speed = sig, None, 0.0, None, 16.7

        def advance(self, t):
            if self.exit_time is not None or t < 10:
                return
            if self.dist is None:
                self.dist = 190.0
                return
            can_go = self.sig.mode.value == "GREEN" and self.sig.active_phase == 3
            nxt = self.dist - 16.7
            if nxt <= 2.0 and not can_go:
                self.dist, self.speed, self.wait = 2.0, 0.0, self.wait + 1.0
            else:
                self.dist, self.speed = nxt, 16.7
                if self.dist <= 0:
                    self.exit_time = t

        def emergency_vehicles(self):
            if self.dist is None or self.exit_time is not None:
                return []
            return [{"id": "amb_0", "approach": "E", "lane": "E2C_2", "dist_to_stop": self.dist,
                     "speed": self.speed, "accum_waiting": self.wait}]

    class Env:
        def __init__(self):
            self.cfg, self.signal, self.sim_time = SimulationConfig(), Sig(), 0.0
            self.observer = Obs(self.signal)
            self.route_summary = NS(ambulances=[{"id": "amb_0", "movement": "left", "approach": "E", "depart": 10.0}])

        def tick(self):
            self.sim_time += 1.0
            self.signal.tick(1.0)
            self.observer.advance(self.sim_time)

    def simulate(enabled: bool):
        env, ctrl = Env(), EmergencyController(FixedTimeController(), enabled=enabled)
        ctrl.reset()
        for _ in range(250):
            if env.signal.mode.value == "GREEN":
                ph = ctrl.select_phase(env)
                if ph != env.signal.active_phase:
                    env.signal.request_phase(ph)
            env.tick()
            ctrl.observe(env)
        return ctrl, env

    on, env_on = simulate(True)
    off, env_off = simulate(False)
    ev_on, ev_off = on.events["amb_0"], off.events["amb_0"]
    print(f"priority ON : cleared at t={ev_on.cleared_time}, green granted t={ev_on.granted_time}, exit t={env_on.observer.exit_time}")
    print(f"priority OFF: cleared at t={ev_off.cleared_time}, exit t={env_off.observer.exit_time}")
    ok = ev_on.status == "cleared" and ev_off.status == "cleared" and ev_on.granted_time is not None
    ok &= env_on.observer.exit_time is not None and env_off.observer.exit_time is not None
    ok &= env_on.observer.exit_time < env_off.observer.exit_time
    ok &= on.mode == EmergencyMode.NORMAL and on.activations == 1
    ok &= required_phase("E", 2) == 3 and required_phase("N", 1) == 0
    print("[ OK ] Emergency controller logic self-test passed" if ok else "[FAIL] self-test failed")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Use --selftest, or run scripts\\run_emergency_demo.py")