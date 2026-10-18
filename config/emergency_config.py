"""Emergency-priority settings and paths."""
from __future__ import annotations

from typing import List, Tuple

from config.settings import DATA_DIR, OUTPUTS_DIR

AMBULANCE_RAW_DIR = DATA_DIR / "ambulance_raw"      # dataset you download (YOLO format)
AMBULANCE_YOLO_DIR = DATA_DIR / "ambulance_yolo"    # prepared single-class dataset (generated)
EMERGENCY_OUT_DIR = OUTPUTS_DIR / "emergency"

# Model class names containing one of these words are treated as "ambulance".
AMBULANCE_NAME_KEYWORDS: Tuple[str, ...] = ("ambulance",)

DEFAULT_DETECTION_RANGE_M = 150.0
DEFAULT_MAX_HOLD_S = 45.0
DEFAULT_CLEARANCE_BUFFER_S = 2.0

# Reproducible demo: (depart_time_s, approach, movement). Duration 600 s.
DEFAULT_SCENARIO: List[Tuple[float, str, str]] = [
    (100.0, "N", "straight"),
    (260.0, "E", "left"),
    (430.0, "W", "straight"),
]