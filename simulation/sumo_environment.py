"""SumoEnvironment: single entry point for fixed-time runs, DQN training/evaluation and the
dashboard. The real SUMO traffic light is controlled through TraCI; nothing is simulated
inside the UI.

Two ways to drive it
--------------------
1. run(controller)            - controller.select_phase(env) is queried every step (fixed-time).
2. reset(); step(action) ...  - RL interface; one step = one decision interval (guarded legality).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from config.constants import (APPROACHES, NUM_PHASES, QUEUE_NORM_VEH, STATE_DIM, WAITING_NORM_S)
from config.settings import GENERATED_DIR, SimulationConfig, ensure_dirs
from evaluation.metrics import MetricsCollector
from simulation.build_network import ensure_network
from simulation.route_generator import RouteSummary, generate_routes
from simulation.traci_controller import (IntersectionObserver, SignalManager, SignalMode,
                                         SimulationError, SumoConnection)
from utils.logging_utils import get_logger

log = get_logger(__name__)


class SumoEnvironment:
    def __init__(self, cfg: SimulationConfig, label: str = "env") -> None:
        cfg.validate()
        self.cfg = cfg
        self.label = label
        self.connection = SumoConnection()
        self.observer: Optional[IntersectionObserver] = None
        self.signal: Optional[SignalManager] = None
        self.metrics = MetricsCollector()
        self.route_summary: Optional[RouteSummary] = None
        self.sim_time = 0.0
        self.last_snapshot: dict = {}

    # ---------------- lifecycle ----------------
    def __enter__(self) -> "SumoEnvironment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """(Re)start SUMO with freshly generated, seeded routes. Returns the initial state."""
        self.close()
        if seed is not None:
            self.cfg = dataclasses.replace(self.cfg, seed=int(seed))
        self.cfg.validate()
        ensure_dirs()
        ensure_network()
        route_path = Path(GENERATED_DIR) / f"routes_{self.label}.rou.xml"
        self.route_summary = generate_routes(self.cfg, route_path)
        self.connection.start(self.cfg, route_path)
        conn = self.connection.conn
        self.observer = IntersectionObserver(conn)
        self.signal = SignalManager(conn, yellow_time=self.cfg.yellow_time,
                                    all_red_time=self.cfg.all_red_time)
        self.signal.initialise(0)
        self.metrics = MetricsCollector()
        self.sim_time = 0.0
        self.last_snapshot = self.observer.lane_snapshot()
        return self.build_state()

    # ---------------- stepping ----------------
    @property
    def done(self) -> bool:
        return self.sim_time >= self.cfg.duration - 1e-9

    def _tick(self) -> None:
        """Advance exactly one simulation step and update signals, observer, metrics."""
        if not self.connection.is_open:
            raise SimulationError("SUMO is not running. Call reset() first.")
        try:
            conn = self.connection.conn
            conn.simulationStep()
            self.sim_time = float(conn.simulation.getTime())
            self.signal.tick(self.cfg.step_length)
            self.observer.step_update(self.sim_time)
            self.last_snapshot = self.observer.lane_snapshot()
            self.metrics.on_step(self.sim_time, self.last_snapshot, self.observer, self.signal.info())
        except SimulationError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise SimulationError(f"SUMO/TraCI failure at t={self.sim_time:.0f}s: {exc}") from exc

    def advance(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.cfg.step_length)))
        for _ in range(n):
            if self.done:
                break
            self._tick()

    def request_phase(self, phase: int) -> bool:
        return self.signal.request_phase(phase)

    # ---------------- RL interface ----------------
    def legalise_action(self, action: int) -> Tuple[int, str]:
        """Apply min/max green rules. Returns (phase_to_apply, override_reason or '')."""
        if not 0 <= action < NUM_PHASES:
            raise ValueError(f"Invalid action {action}; must be 0..{NUM_PHASES - 1}")
        sig, cfg = self.signal, self.cfg
        active = sig.active_phase
        if action != active and sig.elapsed_green < cfg.min_green:
            return active, "min_green"
        if action == active and sig.elapsed_green + cfg.decision_interval > cfg.max_green:
            return (active + 1) % NUM_PHASES, "max_green"
        return action, ""

    def step(self, action: int) -> Tuple[np.ndarray, bool, Dict]:
        """One decision interval. Returns (next_state, done, info). Reward is computed by
        reinforcement_learning.reward from `info` (keeps the environment reward-agnostic)."""
        if self.signal.mode != SignalMode.GREEN:
            raise SimulationError("step() requires the signal to be GREEN (internal error).")
        applied, reason = self.legalise_action(int(action))
        start_idx = len(self.metrics.records)
        switched = applied != self.signal.active_phase
        if switched:
            self.signal.request_phase(applied)
            guard = 0
            while self.signal.mode != SignalMode.GREEN and not self.done and guard < 100:
                self._tick()
                guard += 1
        self.advance(self.cfg.decision_interval)
        agg = self.metrics.interval(start_idx)
        info = {
            "time": self.sim_time, "action_requested": int(action), "action_applied": int(applied),
            "overridden": bool(reason), "override_reason": reason, "switched": switched,
            "interval_queue_mean": agg["queue_mean"], "interval_waiting_mean": agg["waiting_mean"],
            "interval_arrived": agg["arrived"], "interval_stopped_mean": agg["stopped_mean"],
            "queue_by_approach": dict(self.last_snapshot["queue"]),
            "waiting_by_approach": dict(self.last_snapshot["waiting"]),
            "active_phase": self.signal.active_phase, "elapsed_green": self.signal.elapsed_green,
        }
        return self.build_state(), self.done, info

    def build_state(self) -> np.ndarray:
        """17-D normalised state (layout documented in config.constants)."""
        snap = self.last_snapshot
        dens = [snap["density"][a] for a in APPROACHES]
        queue = [min(snap["queue"][a] / QUEUE_NORM_VEH, 1.0) for a in APPROACHES]
        wait = [min(snap["waiting"][a] / WAITING_NORM_S, 1.0) for a in APPROACHES]
        onehot = [1.0 if p == self.signal.active_phase else 0.0 for p in range(NUM_PHASES)]
        elapsed = min(self.signal.elapsed_green / self.cfg.max_green, 1.0)
        state = np.asarray(dens + queue + wait + onehot + [elapsed], dtype=np.float32)
        assert state.shape == (STATE_DIM,), f"state dim {state.shape} != {STATE_DIM}"
        return state

    # ---------------- controller-driven run ----------------
    def run(self, controller, on_step: Optional[Callable[["SumoEnvironment"], None]] = None) -> Dict:
        """Run to the end of the configured duration with a per-step controller. Bounded loop."""
        controller.reset()
        max_iters = int(self.cfg.duration / self.cfg.step_length) + 50
        for _ in range(max_iters):
            if self.done:
                break
            if self.signal.mode == SignalMode.GREEN:
                phase = controller.select_phase(self)
                if phase != self.signal.active_phase:
                    self.signal.request_phase(phase)
            self._tick()
            if on_step is not None:
                on_step(self)
        return self.summary()

    def summary(self) -> Dict:
        return self.metrics.summary(self.sim_time, self.observer)