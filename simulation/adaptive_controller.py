"""DQN-driven adaptive controller for the real SUMO traffic light.

Interface is identical to FixedTimeController (reset / select_phase), so
SumoEnvironment.run(controller) works for both. The signal itself is still driven by
SignalManager (yellow + all-red enforced), so the agent can never create an unsafe state.

Decision timing reproduces training: a decision is made at elapsed_green = 0 and then every
`decision_interval` seconds of green; after a switch the next decision comes
`decision_interval` seconds into the new green.

Logic self-test with a stub environment (no SUMO needed, NOT a traffic result):
    python -m simulation.adaptive_controller --selftest
"""
from __future__ import annotations

import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

from config.constants import (APPROACH_NAMES, APPROACHES, NUM_PHASES, PHASE_GREEN, PHASE_NAMES,
                              QUEUE_NORM_VEH, WAITING_NORM_S)
from config.rl_config import BEST_CKPT, LAST_CKPT
from reinforcement_learning.action_mask import legal_action_mask
from reinforcement_learning.dqn_agent import CheckpointError, DQNAgent
from utils.logging_utils import get_logger

log = get_logger(__name__)

CAVEAT = ("Q-values are the network's learned estimates of long-term return for each phase. "
          "They show which phase the network prefers; they are not a complete causal "
          "explanation of why it prefers it.")


def default_checkpoint_path() -> Path:
    for p in (BEST_CKPT, LAST_CKPT):
        if p.exists():
            return p
    raise CheckpointError("No DQN checkpoint found (models/dqn/best.pt or last.pt).\n"
                          "Train first:  python -m reinforcement_learning.train --episodes 2 --duration 300")


def explain_decision(state: np.ndarray, q: np.ndarray, mask: np.ndarray, requested: int,
                     applied: int, reason: str, active_before: int, elapsed: float) -> Dict:
    """Human-readable summary built ONLY from the state vector and Q-values."""
    queue = {a: float(state[4 + i]) * QUEUE_NORM_VEH for i, a in enumerate(APPROACHES)}
    wait = {a: float(state[8 + i]) * WAITING_NORM_S for i, a in enumerate(APPROACHES)}
    pressure = {PHASE_NAMES[p]: sum(queue[a] for a in PHASE_GREEN[p]) for p in range(NUM_PHASES)}
    ranking = [{"phase": p, "name": PHASE_NAMES[p], "q": float(q[p]), "legal": bool(mask[p])}
               for p in np.argsort(-q)]
    legal_sorted = [r for r in ranking if r["legal"]]
    margin = (legal_sorted[0]["q"] - legal_sorted[1]["q"]) if len(legal_sorted) > 1 else None
    busiest = sorted(queue.items(), key=lambda kv: -kv[1])[:2]
    top_pressure = max(pressure, key=pressure.get)

    kind = "hold" if applied == active_before else "switch to"
    lines = [f"Active phase before decision: {PHASE_NAMES[active_before]} ({elapsed:.0f} s green so far).",
             "Longest queues (approx. vehicles): "
             + ", ".join(f"{APPROACH_NAMES[a]} {v:.0f}" for a, v in busiest) + ".",
             f"Legal actions: {[PHASE_NAMES[i] for i in range(NUM_PHASES) if mask[i]]}.",
             f"Decision: {kind} {PHASE_NAMES[applied]} (Q = {q[applied]:.2f})."]
    if margin is not None:
        lines.append(f"Margin over next-best legal action: {margin:.2f}.")
    if reason:
        lines.append(f"Safety rule '{reason}' replaced the requested action "
                     f"{PHASE_NAMES[requested]} with {PHASE_NAMES[applied]}.")
    return {"summary": lines[3], "lines": lines, "queue_by_approach": queue,
            "waiting_by_approach": wait, "phase_pressure": pressure, "ranking": ranking,
            "margin": margin, "agrees_with_queue_pressure": PHASE_NAMES[applied] == top_pressure,
            "caveat": CAVEAT}


@dataclass
class DecisionRecord:
    time: float
    state: List[float]
    q_values: List[float]
    mask: List[bool]
    requested: int
    applied: int
    override_reason: str
    active_before: int
    elapsed_green: float
    explanation: Dict

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptiveController:
    name = "dqn"

    def __init__(self, agent: DQNAgent, checkpoint_path: Optional[Path] = None,
                 history_size: int = 500) -> None:
        self.agent = agent
        self.agent.policy.eval()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.history: Deque[DecisionRecord] = deque(maxlen=history_size)
        self.reset()

    @classmethod
    def from_checkpoint(cls, path: Optional[Path] = None, device: str = "cpu") -> "AdaptiveController":
        p = Path(path) if path else default_checkpoint_path()
        agent = DQNAgent.from_checkpoint(p, device=device, with_buffer=False)
        return cls(agent, p)

    # ---------------- lifecycle ----------------
    def reset(self) -> None:
        self._next_elapsed = 0.0
        self._last_phase: Optional[int] = None
        self.history.clear()
        self.last: Optional[DecisionRecord] = None
        self.decisions = 0
        self.overrides = 0
        self.switches = 0
        self.phase_usage: Counter = Counter()

    # ---------------- control ----------------
    def select_phase(self, env) -> int:
        """Called every step while the signal is GREEN; decides only at decision moments."""
        sig, cfg = env.signal, env.cfg
        if self._last_phase is not None and sig.active_phase != self._last_phase:
            self._next_elapsed = float(cfg.decision_interval)   # phase changed (by us or externally)
        self._last_phase = sig.active_phase
        if sig.elapsed_green + 1e-9 < self._next_elapsed:
            return sig.active_phase

        state = env.build_state()
        mask = legal_action_mask(env)
        q = self.agent.q_values(state)
        requested = self.agent.select_action(state, mask, greedy=True)
        applied, reason = env.legalise_action(requested)
        active_before, elapsed = sig.active_phase, float(sig.elapsed_green)

        rec = DecisionRecord(
            time=float(getattr(env, "sim_time", 0.0)), state=[float(x) for x in state],
            q_values=[float(x) for x in q], mask=[bool(m) for m in mask], requested=int(requested),
            applied=int(applied), override_reason=reason, active_before=int(active_before),
            elapsed_green=elapsed,
            explanation=explain_decision(state, q, mask, requested, applied, reason,
                                         active_before, elapsed))
        self.history.append(rec)
        self.last = rec
        self.decisions += 1
        self.overrides += int(bool(reason))
        self.phase_usage[PHASE_NAMES[applied]] += 1
        if applied != active_before:
            self.switches += 1
            self._next_elapsed = float(cfg.decision_interval)
        else:
            self._next_elapsed = elapsed + float(cfg.decision_interval)
        return int(applied)

    # ---------------- information ----------------
    def stats(self) -> Dict:
        return {"decisions": self.decisions, "overrides": self.overrides, "switches": self.switches,
                "phase_usage": dict(self.phase_usage)}

    def info(self) -> Dict:
        meta = self.agent.meta or {}
        return {"checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
                "episodes_trained": self.agent.episodes, "env_steps": self.agent.env_steps,
                "learn_steps": self.agent.learn_steps, "checkpoint_epsilon": self.agent.epsilon,
                "epsilon_in_use": 0.0, "trained_at": meta.get("trained_at"),
                "best_val_wait_s": meta.get("best_val_wait_s"),
                "demand_range": meta.get("demand_range"),
                "hidden": list(self.agent.config.hidden), "double_dqn": self.agent.config.double_dqn,
                "parameters": self.agent.policy.n_parameters()}

    def compatibility_warnings(self, cfg) -> List[str]:
        """Compare run settings with the conditions the checkpoint was trained under."""
        meta = self.agent.meta or {}
        tc = meta.get("train_sim_config")
        if not tc:
            return ["Checkpoint has no training-config metadata; compatibility cannot be checked."]
        out: List[str] = []
        for k in ("min_green", "max_green", "decision_interval", "yellow_time", "all_red_time",
                  "step_length"):
            if k in tc and getattr(cfg, k) != tc[k]:
                out.append(f"{k} = {getattr(cfg, k)} differs from training value {tc[k]}.")
        dr = meta.get("demand_range")
        if dr:
            lo, hi = float(dr[0]), float(dr[1])
            bad = {a: v for a, v in cfg.demand_vph.items() if not lo <= v <= hi}
            if bad:
                out.append(f"Demand {bad} veh/h is outside the training range {lo:.0f}-{hi:.0f} "
                           "veh/h; the policy is not evidenced there.")
        return out


# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Checks decision timing and legality with a stub env and an UNTRAINED network."""
    from config.settings import SimulationConfig
    from reinforcement_learning.dqn_agent import DQNConfig
    from simulation.sumo_environment import SumoEnvironment

    class Sig:
        active_phase = 0
        elapsed_green = 0.0

    class Stub:
        cfg = SimulationConfig()
        signal = Sig()
        sim_time = 0.0
        _rng = np.random.default_rng(0)

        def build_state(self):
            return self._rng.random(17).astype(np.float32)

        def legalise_action(self, action):
            return SumoEnvironment.legalise_action(self, action)

    env = Stub()
    ctrl = AdaptiveController(DQNAgent(DQNConfig(seed=0), with_buffer=False))
    ctrl.reset()
    cfg = env.cfg
    log_rows, ok = [], True
    for _tick in range(400):
        before_n, before_phase, before_el = ctrl.decisions, env.signal.active_phase, env.signal.elapsed_green
        phase = ctrl.select_phase(env)
        if ctrl.decisions > before_n:
            log_rows.append((before_el, before_phase, phase))
        ok &= 0 <= phase < NUM_PHASES
        if phase != env.signal.active_phase:            # emulate a completed switch (yellow skipped)
            env.signal.active_phase, env.signal.elapsed_green = phase, 0.0
        env.signal.elapsed_green += 1.0
        env.sim_time += 1.0
    ok &= bool(log_rows) and log_rows[0][0] == 0.0
    for el, before, after in log_rows:
        if after != before and el < cfg.min_green:
            ok = False
            print(f"[FAIL] switched at elapsed {el} < min_green")
        if after == before and el + cfg.decision_interval > cfg.max_green:
            ok = False
            print(f"[FAIL] held at elapsed {el} beyond max_green")
    ok &= len(log_rows) == ctrl.decisions
    print(f"decisions: {ctrl.decisions}, switches: {ctrl.switches}, overrides: {ctrl.overrides}")
    print("[ OK ] Adaptive controller timing/legality logic passed" if ok else "[FAIL] self-test failed")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Use --selftest, or run scripts\\run_comparison.py")