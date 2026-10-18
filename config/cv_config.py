"""Computer-vision configuration: paths, class mappings, colours."""
from __future__ import annotations

from typing import Dict, Tuple

from config.settings import DATA_DIR, MODELS_DIR, OUTPUTS_DIR

HF_REPO_ID = "kalyan1729/sentineledgedataset"

RAW_DIR = DATA_DIR / "detrac_raw"            # downloaded images + XML
YOLO_DIR = DATA_DIR / "detrac_yolo"          # converted YOLO dataset (generated)
VISION_OUT_DIR = OUTPUTS_DIR / "vision"
REGION_CONFIG_DIR = DATA_DIR / "config"
LANE_REGION_FILE = REGION_CONFIG_DIR / "lane_regions.json"

VEHICLE_WEIGHTS = MODELS_DIR / "vehicle_detector.pt"   # produced by Colab training
PRETRAINED_NAME = "yolov8n.pt"
PRETRAINED_WEIGHTS = MODELS_DIR / PRETRAINED_NAME      # COCO-pretrained fallback
AMBULANCE_WEIGHTS = MODELS_DIR / "ambulance_detector.pt"  # Part 6

SEQUENCE_REGEX = r"MVI_\d+"

# Classes reported by the system (what the rest of the project understands).
UNIFIED_CLASSES: Tuple[str, ...] = ("car", "motorcycle", "bus", "truck")

# Model class name (lower case) -> unified class. Used for BOTH the COCO model and a
# DETRAC-fine-tuned model.
#   * "van" -> car      : vans are light vehicles (assumption, edit if you disagree)
#   * "others" -> truck : DETRAC "others" is a heterogeneous group (mostly trucks / large
#                         vehicles). This is an ASSUMPTION; edit it if your data says otherwise.
MODEL_NAME_TO_UNIFIED: Dict[str, str] = {
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "van": "car",
    "others": "truck",
}

DETRAC_PREFERRED_ORDER: Tuple[str, ...] = ("car", "van", "bus", "others")

CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "car": (255, 170, 0),
    "motorcycle": (0, 200, 255),
    "bus": (120, 200, 0),
    "truck": (255, 120, 170),
    "ambulance": (60, 40, 255),
}
DEFAULT_COLOR_BGR = (200, 200, 200)