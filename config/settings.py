"""Paths, SUMO discovery and the validated SimulationConfig used everywhere."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.constants import APPROACHES, MOVEMENTS, VEHICLE_TYPES

try:  # optional
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

SUMO_CONFIG_DIR = PROJECT_ROOT / "sumo_config"
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

NET_FILE = SUMO_CONFIG_DIR / "intersection.net.xml"
SUMOCFG_FILE = SUMO_CONFIG_DIR / "simulation.sumocfg"
DEFAULT_ROUTE_FILE = SUMO_CONFIG_DIR / "traffic.rou.xml"

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


def ensure_dirs() -> None:
    for d in (SUMO_CONFIG_DIR, DATA_DIR, GENERATED_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# SUMO discovery
# --------------------------------------------------------------------------- #
SUMO_SETUP_HELP = """
SUMO could not be found. Fix with ONE of these options (PowerShell):

  Option A (recommended, no installer needed):
      .\\.venv\\Scripts\\Activate.ps1
      pip install eclipse-sumo traci sumolib

  Option B (stand-alone installer with sumo-gui):
      1. Download the MSI from https://sumo.dlr.de/docs/Downloads.php and install it.
      2. [Environment]::SetEnvironmentVariable("SUMO_HOME","C:\\Program Files (x86)\\Eclipse\\Sumo","User")
      3. Close and reopen PowerShell / VS Code.

Then verify with:  python scripts\\check_environment.py
""".strip()


class SumoNotFoundError(RuntimeError):
    """Raised when SUMO binaries or the TraCI Python API cannot be located."""


def find_sumo_home() -> Optional[Path]:
    env = os.environ.get("SUMO_HOME")
    if env and Path(env).exists():
        return Path(env)
    try:  # pip package "eclipse-sumo"
        import sumo  # type: ignore

        home = getattr(sumo, "SUMO_HOME", None)
        if home and Path(home).exists():
            return Path(home)
        return Path(sumo.__file__).resolve().parent
    except Exception:
        return None


def find_sumo_binary(name: str) -> str:
    """Return the path of a SUMO executable such as 'sumo', 'sumo-gui', 'netconvert'."""
    exe = name + (".exe" if os.name == "nt" else "")
    found = shutil.which(name)
    if found:
        return found
    candidates: List[Path] = [Path(sys.executable).resolve().parent / exe]
    home = find_sumo_home()
    if home is not None:
        candidates.append(home / "bin" / exe)
    for c in candidates:
        if c.exists():
            return str(c)
    raise SumoNotFoundError(f"Executable '{name}' not found.\n\n{SUMO_SETUP_HELP}")


def ensure_traci_importable() -> None:
    """Make sure `import traci` works (falls back to $SUMO_HOME/tools)."""
    try:
        import traci  # noqa: F401
        return
    except ImportError:
        home = find_sumo_home()
        if home is not None and (home / "tools").exists():
            sys.path.append(str(home / "tools"))
            try:
                import traci  # noqa: F401
                return
            except ImportError:
                pass
    raise SumoNotFoundError(f"Python package 'traci' not importable.\n\n{SUMO_SETUP_HELP}")


# --------------------------------------------------------------------------- #
# Simulation configuration
# --------------------------------------------------------------------------- #
def _default_demand() -> Dict[str, float]:
    return {a: 500.0 for a in APPROACHES}


@dataclass
class SimulationConfig:
    """All user-tunable simulation parameters (validated)."""

    duration: int = 1800                    # simulated seconds
    step_length: float = 1.0                # seconds per TraCI step
    seed: int = 42
    demand_vph: Dict[str, float] = field(default_factory=_default_demand)  # veh/h per approach
    turn_probs: Dict[str, float] = field(
        default_factory=lambda: {"right": 0.2, "straight": 0.6, "left": 0.2})
    vehicle_mix: Dict[str, float] = field(
        default_factory=lambda: {"car": 0.70, "motorcycle": 0.12, "bus": 0.06, "truck": 0.12})
    max_vehicles: Optional[int] = None      # cap on total generated vehicles
    emergency_probability: float = 0.0      # fraction of vehicles replaced by ambulances
    # explicit scheduled ambulances: (depart_time_s, approach, movement)
    emergency_events: List[Tuple[float, str, str]] = field(default_factory=list)

    min_green: int = 10
    max_green: int = 60
    yellow_time: int = 3
    all_red_time: int = 1
    decision_interval: int = 5              # RL decision period (s)

    gui: bool = False

    @classmethod
    def with_uniform_demand(cls, vph: float, **kwargs) -> "SimulationConfig":
        cfg = cls(**kwargs)
        cfg.demand_vph = {a: float(vph) for a in APPROACHES}
        return cfg

    def validate(self) -> None:
        e: List[str] = []
        if not (60 <= self.duration <= 86400):
            e.append("duration must be between 60 and 86400 seconds")
        if not (0.1 <= self.step_length <= 1.0):
            e.append("step_length must be within 0.1 .. 1.0 seconds")
        if not isinstance(self.seed, int) or self.seed < 0:
            e.append("seed must be a non-negative integer")
        for a in APPROACHES:
            v = self.demand_vph.get(a)
            if v is None or not (0 <= v <= 3000):
                e.append(f"demand_vph[{a}] must be within 0..3000 veh/h")
        if set(self.turn_probs) != set(MOVEMENTS) or abs(sum(self.turn_probs.values()) - 1) > 1e-3 \
                or any(p < 0 for p in self.turn_probs.values()):
            e.append("turn_probs must have keys right/straight/left, be >=0 and sum to 1")
        if not set(self.vehicle_mix) <= set(VEHICLE_TYPES) or abs(sum(self.vehicle_mix.values()) - 1) > 1e-3 \
                or any(p < 0 for p in self.vehicle_mix.values()):
            e.append(f"vehicle_mix keys must be within {VEHICLE_TYPES}, be >=0 and sum to 1")
        if self.max_vehicles is not None and self.max_vehicles < 1:
            e.append("max_vehicles must be >= 1 or None")
        if not (0.0 <= self.emergency_probability <= 0.2):
            e.append("emergency_probability must be within 0..0.2")
        for ev in self.emergency_events:
            try:
                t, a, m = ev
                if t < 0 or t >= self.duration or a not in APPROACHES or m not in MOVEMENTS:
                    raise ValueError
            except (TypeError, ValueError):
                e.append(f"invalid emergency event {ev!r}: expected (time, approach, movement)")
        if self.min_green < 3:
            e.append("min_green must be >= 3 s")
        if self.max_green <= self.min_green:
            e.append("max_green must be greater than min_green")
        if self.yellow_time < 1:
            e.append("yellow_time must be >= 1 s")
        if self.all_red_time < 0:
            e.append("all_red_time must be >= 0 s")
        if not (1 <= self.decision_interval <= self.min_green):
            e.append("decision_interval must be within 1..min_green")
        if e:
            raise ValueError("Invalid simulation settings:\n - " + "\n - ".join(e))

    def to_dict(self) -> dict:
        return asdict(self)