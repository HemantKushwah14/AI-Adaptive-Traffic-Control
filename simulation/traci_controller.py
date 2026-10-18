"""TraCI layer: process/connection management, safe signal state machine, and observation.

Safety design of the signal machine
-----------------------------------
A phase change is ALWAYS: GREEN(old) -> YELLOW(old greens only) -> ALL_RED -> GREEN(new).
Only the four conflict-free phases in config.constants.PHASE_GREEN can ever be shown, so an
illegal / conflicting signal state cannot be produced by any controller.
"""
from __future__ import annotations

import itertools
import os
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from config.constants import (APPROACHES, DENSITY_CAPACITY_VEH, NUM_PHASES, PHASE_GREEN,
                              PHASE_NAMES, TLS_ID)
from config.settings import (NET_FILE, SUMOCFG_FILE, SimulationConfig, ensure_traci_importable,
                             find_sumo_binary)
from utils.logging_utils import get_logger

log = get_logger(__name__)


class SimulationError(RuntimeError):
    """Raised when SUMO/TraCI fails during a run."""


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
class SumoConnection:
    """Owns one SUMO subprocess + its TraCI connection (unique label => many sessions OK)."""

    _counter = itertools.count()

    def __init__(self) -> None:
        self.conn = None
        self.label: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.conn is not None

    def start(self, cfg: SimulationConfig, route_file: Path) -> None:
        ensure_traci_importable()
        import traci

        if not NET_FILE.exists() or not SUMOCFG_FILE.exists():
            raise FileNotFoundError("SUMO network missing. Run:  python -m simulation.build_network")
        if not Path(route_file).exists():
            raise FileNotFoundError(f"Route file missing: {route_file}")
        self.close()

        binary = find_sumo_binary("sumo-gui" if cfg.gui else "sumo")
        cmd = [binary, "-c", str(SUMOCFG_FILE), "--route-files", str(route_file),
               "--seed", str(cfg.seed), "--step-length", str(cfg.step_length),
               "--end", str(cfg.duration + 5), "--no-step-log", "true", "--no-warnings", "true",
               "--time-to-teleport", "-1", "--waiting-time-memory", str(int(cfg.duration) + 100),
               "--duration-log.disable", "true"]
        if cfg.gui:
            cmd += ["--start", "--quit-on-end"]
        self.label = f"sim_{os.getpid()}_{next(self._counter)}_{int(time.time() * 1000)}"
        try:
            traci.start(cmd, label=self.label)
            self.conn = traci.getConnection(self.label)
        except Exception as exc:
            self.conn = None
            raise SimulationError(f"Could not start SUMO/TraCI: {exc}") from exc
        log.info("SUMO started (label=%s, seed=%d, gui=%s)", self.label, cfg.seed, cfg.gui)

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception as exc:  # already closed / process died
                log.debug("TraCI close: %s", exc)
            self.conn = None
            log.info("SUMO connection closed (%s)", self.label)


# --------------------------------------------------------------------------- #
# Signal state machine
# --------------------------------------------------------------------------- #
class SignalMode(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


class SignalManager:
    """Drives the real SUMO traffic light through setRedYellowGreenState."""

    def __init__(self, conn, tls_id: str = TLS_ID, yellow_time: float = 3.0,
                 all_red_time: float = 1.0) -> None:
        self.conn = conn
        self.tls_id = tls_id
        self.yellow_time = float(yellow_time)
        self.all_red_time = float(all_red_time)
        self._signals: List[tuple] = []  # per signal index: (approach, lane_index)
        self._build_index_map()

        self.active_phase = 0
        self.target_phase = 0
        self.mode = SignalMode.GREEN
        self.elapsed_green = 0.0
        self._timer = 0.0
        self.phase_switches = 0

    # -- setup --
    def _build_index_map(self) -> None:
        groups = self.conn.trafficlight.getControlledLinks(self.tls_id)
        for i, group in enumerate(groups):
            if not group:
                raise SimulationError(f"Signal index {i} of '{self.tls_id}' controls no link.")
            in_lane = group[0][0]                       # e.g. "N2C_1"
            approach = in_lane[0]
            lane_idx = int(in_lane.rsplit("_", 1)[1])
            if approach not in APPROACHES:
                raise SimulationError(f"Unexpected incoming lane '{in_lane}'.")
            self._signals.append((approach, lane_idx))
        covered = {s for p in PHASE_GREEN.values() for a, lanes in p.items() for s in ((a, l) for l in lanes)}
        if set(self._signals) != covered:
            raise SimulationError("Signal links do not match the phase definition. "
                                  "Rebuild the network: python -m simulation.build_network --force")

    def initialise(self, phase: int = 0) -> None:
        self.active_phase = self.target_phase = phase
        self.mode = SignalMode.GREEN
        self.elapsed_green = 0.0
        self._apply(self.green_state(phase))

    # -- state strings --
    def green_state(self, phase: int) -> str:
        g = PHASE_GREEN[phase]
        return "".join("G" if lane in g.get(a, ()) else "r" for a, lane in self._signals)

    def yellow_state(self, phase: int) -> str:
        g = PHASE_GREEN[phase]
        return "".join("y" if lane in g.get(a, ()) else "r" for a, lane in self._signals)

    def all_red_state(self) -> str:
        return "r" * len(self._signals)

    def _apply(self, state: str) -> None:
        self.conn.trafficlight.setRedYellowGreenState(self.tls_id, state)
        self.current_state = state

    # -- control --
    def request_phase(self, phase: int) -> bool:
        """Request a phase. Returns True if a transition started / target was updated."""
        if not 0 <= phase < NUM_PHASES:
            raise ValueError(f"Invalid phase {phase}; must be 0..{NUM_PHASES - 1}")
        if self.mode == SignalMode.GREEN:
            if phase == self.active_phase:
                return False
            self.target_phase = phase
            self.mode = SignalMode.YELLOW
            self._timer = self.yellow_time
            self.phase_switches += 1
            self._apply(self.yellow_state(self.active_phase))
            return True
        self.target_phase = phase  # mid-transition: safe, new target is shown after all-red
        return True

    def tick(self, dt: float) -> None:
        """Advance the state machine by dt seconds (call once per simulation step)."""
        if self.mode == SignalMode.GREEN:
            self.elapsed_green += dt
            self._apply(self.green_state(self.active_phase))
        elif self.mode == SignalMode.YELLOW:
            self._timer -= dt
            if self._timer <= 1e-9:
                if self.all_red_time > 0:
                    self.mode = SignalMode.ALL_RED
                    self._timer = self.all_red_time
                    self._apply(self.all_red_state())
                else:
                    self._activate_target()
            else:
                self._apply(self.yellow_state(self.active_phase))
        else:  # ALL_RED
            self._timer -= dt
            if self._timer <= 1e-9:
                self._activate_target()
            else:
                self._apply(self.all_red_state())

    def _activate_target(self) -> None:
        self.active_phase = self.target_phase
        self.mode = SignalMode.GREEN
        self.elapsed_green = 0.0
        self._timer = 0.0
        self._apply(self.green_state(self.active_phase))

    # -- info --
    @property
    def transition_remaining(self) -> float:
        return max(self._timer, 0.0) if self.mode != SignalMode.GREEN else 0.0

    def info(self) -> dict:
        return {
            "active_phase": self.active_phase,
            "target_phase": self.target_phase,
            "phase_name": PHASE_NAMES[self.active_phase],
            "mode": self.mode.value,
            "elapsed_green": self.elapsed_green,
            "transition_remaining": self.transition_remaining,
            "green_approaches": sorted(PHASE_GREEN[self.active_phase].keys())
            if self.mode == SignalMode.GREEN else [],
            "state_string": getattr(self, "current_state", ""),
            "phase_switches": self.phase_switches,
        }


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
class IntersectionObserver:
    """Reads lane/vehicle state from SUMO. One vehicle subscription gives metrics AND
    positions for the dashboard (no extra per-vehicle calls)."""

    def __init__(self, conn, tls_id: str = TLS_ID) -> None:
        ensure_traci_importable()
        import traci.constants as tc

        self.conn = conn
        self.tc = tc
        self._vars = [tc.VAR_SPEED, tc.VAR_WAITING_TIME, tc.VAR_ACCUMULATED_WAITING_TIME,
                      tc.VAR_TIMELOSS, tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_TYPE,
                      tc.VAR_LANE_ID, tc.VAR_LANEPOSITION]

        lanes = list(dict.fromkeys(conn.trafficlight.getControlledLanes(tls_id)))
        self.lanes_by_approach: Dict[str, List[str]] = {a: [] for a in APPROACHES}
        for lane in lanes:
            self.lanes_by_approach[lane[0]].append(lane)
        self.lane_length = {lane: conn.lane.getLength(lane) for lane in lanes}

        self.vehicles: Dict[str, dict] = {}
        self.depart_time: Dict[str, float] = {}
        self.finished: List[dict] = []
        self.newly_finished: List[dict] = []
        self.newly_departed: List[str] = []
        self._last: Dict[str, dict] = {}

    # -- vehicles --
    def _parse(self, vid: str, res: dict) -> dict:
        tc = self.tc
        pos = res.get(tc.VAR_POSITION, (0.0, 0.0))
        return {
            "id": vid, "x": float(pos[0]), "y": float(pos[1]),
            "angle": float(res.get(tc.VAR_ANGLE, 0.0)), "speed": float(res.get(tc.VAR_SPEED, 0.0)),
            "waiting": float(res.get(tc.VAR_WAITING_TIME, 0.0)),
            "accum_waiting": float(res.get(tc.VAR_ACCUMULATED_WAITING_TIME, 0.0)),
            "timeloss": float(res.get(tc.VAR_TIMELOSS, 0.0)),
            "type": res.get(tc.VAR_TYPE, "car"), "lane": res.get(tc.VAR_LANE_ID, ""),
            "lane_pos": float(res.get(tc.VAR_LANEPOSITION, 0.0)),
        }

    def step_update(self, sim_time: float) -> None:
        """Call once after every simulationStep()."""
        c = self.conn
        self.newly_departed = list(c.simulation.getDepartedIDList())
        for vid in self.newly_departed:
            c.vehicle.subscribe(vid, self._vars)
            self.depart_time[vid] = sim_time

        self.newly_finished = []
        for vid in c.simulation.getArrivedIDList():
            last = self._last.pop(vid, None)
            dep = self.depart_time.pop(vid, sim_time)
            rec = {"id": vid, "travel_time": sim_time - dep,
                   "waiting": last["accum_waiting"] if last else 0.0,
                   "timeloss": last["timeloss"] if last else 0.0,
                   "type": last["type"] if last else "car"}
            self.newly_finished.append(rec)
            self.finished.append(rec)

        raw = c.vehicle.getAllSubscriptionResults()
        self.vehicles = {vid: self._parse(vid, res) for vid, res in raw.items() if res}
        self._last.update(self.vehicles)

    def vehicle_list(self) -> List[dict]:
        return list(self.vehicles.values())

    def emergency_vehicles(self) -> List[dict]:
        """Ambulances currently in the network, with approach and distance to stop line."""
        out = []
        for v in self.vehicles.values():
            if v["type"] != "ambulance":
                continue
            lane = v["lane"]
            on_incoming = len(lane) > 3 and lane[1:3] == "2C" and lane[0] in APPROACHES
            approach = lane[0] if on_incoming else None
            dist = self.lane_length.get(lane, 0.0) - v["lane_pos"] if on_incoming else None
            out.append({**v, "approach": approach, "dist_to_stop": dist})
        return out

    # -- lanes --
    def lane_snapshot(self) -> dict:
        c = self.conn
        queue, waiting, count, density = {}, {}, {}, {}
        lanes_detail = []
        for a in APPROACHES:
            q = w = n = 0.0
            for lane in self.lanes_by_approach[a]:
                lq = c.lane.getLastStepHaltingNumber(lane)
                lw = c.lane.getWaitingTime(lane)
                ln = c.lane.getLastStepVehicleNumber(lane)
                q, w, n = q + lq, w + lw, n + ln
                lanes_detail.append({"lane": lane, "approach": a,
                                     "index": int(lane.rsplit("_", 1)[1]), "queue": lq,
                                     "count": ln, "occupancy": c.lane.getLastStepOccupancy(lane)})
            queue[a], waiting[a], count[a] = q, w, n
            density[a] = min(n / DENSITY_CAPACITY_VEH, 1.0)
        return {"queue": queue, "waiting": waiting, "count": count, "density": density,
                "queue_total": sum(queue.values()), "waiting_total": sum(waiting.values()),
                "count_total": sum(count.values()), "lanes": lanes_detail}