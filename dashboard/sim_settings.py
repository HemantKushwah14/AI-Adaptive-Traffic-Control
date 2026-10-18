"""Dashboard simulation settings: defaults, SimulationConfig builder and controller factory.
Streamlit-free on purpose (used by the headless smoke test and tests)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from config.constants import APPROACHES
from config.rl_config import BEST_CKPT, LAST_CKPT
from config.settings import SimulationConfig
from simulation.adaptive_controller import AdaptiveController
from simulation.emergency_controller import EmergencyController
from simulation.fixed_time_controller import FixedTimeController

CONTROLLERS = {"fixed_time": "Fixed-time (baseline)", "dqn": "DQN (adaptive)"}
SPEED_OPTIONS = {"1×": 1.0, "2×": 2.0, "5×": 5.0, "10×": 10.0, "20×": 20.0, "Max": 0.0}


def default_settings() -> Dict[str, Any]:
    have_ckpt = BEST_CKPT.exists() or LAST_CKPT.exists()
    return {
        "demand_vph": {a: 500.0 for a in APPROACHES},
        "duration": 900,
        "seed": 42,
        "step_length": 1.0,
        "min_green": 10,
        "max_green": 60,
        "emergency_probability": 0.0,
        "emergency_events": [],            # [(time_s, approach, movement)]
        "max_vehicles": None,
        "controller": "dqn" if have_ckpt else "fixed_time",
        "emergency_priority": True,
        "checkpoint": None,                # None -> models/dqn/best.pt, else last.pt
        "fixed_greens": (35.0, 10.0, 35.0, 10.0),
        "speed": 5.0,
    }


def build_sim_config(settings: Dict[str, Any]) -> SimulationConfig:
    s = settings
    cfg = SimulationConfig(
        duration=int(s["duration"]), step_length=float(s["step_length"]), seed=int(s["seed"]),
        demand_vph={a: float(s["demand_vph"][a]) for a in APPROACHES},
        max_vehicles=s.get("max_vehicles"),
        emergency_probability=float(s.get("emergency_probability", 0.0)),
        emergency_events=[(float(t), str(a), str(m)) for t, a, m in s.get("emergency_events", [])],
        min_green=int(s["min_green"]), max_green=int(s["max_green"]),
        decision_interval=min(5, int(s["min_green"])))
    cfg.validate()
    return cfg


def make_controller(settings: Dict[str, Any], cfg: SimulationConfig
                    ) -> Tuple[Any, EmergencyController, List[str]]:
    """Returns (base_controller, emergency_wrapper, compatibility_warnings)."""
    kind = settings.get("controller", "fixed_time")
    warns: List[str] = []
    if kind == "dqn":
        ck = settings.get("checkpoint")
        base = AdaptiveController.from_checkpoint(Path(ck) if ck else None)
        warns = base.compatibility_warnings(cfg)
    elif kind == "fixed_time":
        base = FixedTimeController(tuple(settings.get("fixed_greens", (35, 10, 35, 10))))
    else:
        raise ValueError(f"Unknown controller '{kind}'")
    ctrl = EmergencyController(base, enabled=bool(settings.get("emergency_priority", True)))
    return base, ctrl, warns