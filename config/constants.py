"""Project-wide constants shared by simulation, RL, vision and dashboard.

Intersection geometry
---------------------
Four approaches N, S, E, W. Each approach has 3 incoming lanes
(lane index 0 = rightmost, as in SUMO):
    0 -> right turn, 1 -> straight, 2 -> left turn.
SUMO's default is right-hand traffic.

Signal phases (4 legal phases, all conflict-free)
-------------------------------------------------
    0 NS_THROUGH : N,S lanes 0,1 (right + straight)
    1 NS_LEFT    : N,S lane 2    (protected left)
    2 EW_THROUGH : E,W lanes 0,1
    3 EW_LEFT    : E,W lane 2
"""
from __future__ import annotations

from typing import Dict, Tuple

APPROACHES: Tuple[str, ...] = ("N", "S", "E", "W")
APPROACH_NAMES: Dict[str, str] = {"N": "North", "S": "South", "E": "East", "W": "West"}

TLS_ID = "C"
NUM_LANES = 3
MOVEMENTS: Tuple[str, ...] = ("right", "straight", "left")
LANE_INDEX: Dict[str, int] = {"right": 0, "straight": 1, "left": 2}

INCOMING_EDGE: Dict[str, str] = {a: f"{a}2C" for a in APPROACHES}
OUTGOING_EDGE: Dict[str, str] = {a: f"C2{a}" for a in APPROACHES}

# Destination arm for each (origin arm, movement), right-hand traffic.
TURN_TARGET: Dict[str, Dict[str, str]] = {
    "N": {"right": "W", "straight": "S", "left": "E"},
    "S": {"right": "E", "straight": "N", "left": "W"},
    "E": {"right": "N", "straight": "W", "left": "S"},
    "W": {"right": "S", "straight": "E", "left": "N"},
}

# ---- Signal phases ----
PHASE_NAMES: Tuple[str, ...] = ("NS_THROUGH", "NS_LEFT", "EW_THROUGH", "EW_LEFT")
NUM_PHASES = len(PHASE_NAMES)
# phase -> {approach: tuple of lane indices that are green}
PHASE_GREEN: Dict[int, Dict[str, Tuple[int, ...]]] = {
    0: {"N": (0, 1), "S": (0, 1)},
    1: {"N": (2,), "S": (2,)},
    2: {"E": (0, 1), "W": (0, 1)},
    3: {"E": (2,), "W": (2,)},
}

# ---- Geometry (metres) ----
ARM_LENGTH_M = 200.0
SPEED_LIMIT_MS = 13.89  # 50 km/h

# ---- Vehicle types ----
VEHICLE_TYPES: Tuple[str, ...] = ("car", "motorcycle", "bus", "truck")
EMERGENCY_TYPE = "ambulance"

# ---- RL state layout (STATE_DIM must match the DQN input size) ----
#  [0:4]   density per approach (N,S,E,W), 0..1
#  [4:8]   queue length per approach, 0..1
#  [8:12]  waiting time per approach, 0..1
#  [12:16] one-hot active phase
#  [16]    elapsed green / max green, 0..1
STATE_DIM = 17
ACTION_DIM = NUM_PHASES

# ---- State normalisation ----
DENSITY_CAPACITY_VEH = 40.0   # vehicles on an approach regarded as "full"
QUEUE_NORM_VEH = 30.0         # halting vehicles regarded as "very long queue"
WAITING_NORM_S = 1000.0       # summed waiting seconds on an approach regarded as "very high"