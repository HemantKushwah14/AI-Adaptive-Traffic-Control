"""Ambulance detection from images/video using a SEPARATELY TRAINED YOLO model.

There is deliberately NO fallback model: COCO has no ambulance class and relabelling
trucks/vans would be fabricated output. If models/ambulance_detector.pt is missing, an
explicit error explains how to create it (notebooks/train_ambulance_colab.ipynb).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from computer_vision.detector import Detection
from computer_vision.lane_analyzer import LaneAnalyzer
from config.cv_config import AMBULANCE_WEIGHTS
from config.emergency_config import AMBULANCE_NAME_KEYWORDS
from utils.logging_utils import get_logger

log = get_logger(__name__)

LIMITATIONS = (
    "Visual ambulance detection is error-prone: look-alike vehicles cause false positives; "
    "unseen liveries, night/rain, occlusion and distance cause misses.",
    "The model does not hear sirens or see flashing lights unless they are visible in the image.",
    "Accuracy only applies to data similar to the training set (camera, country, livery).",
    "Use it as an assistive trigger with multi-frame confirmation, not as a safety-critical sensor.",
)

HELP = (
    "Ambulance model not found: {path}\n"
    "1. Download an ambulance dataset in YOLO format (see README / Part 6) into data\\ambulance_raw\n"
    "2. python -m dataset.prepare_ambulance_dataset --list-classes\n"
    "3. Train with notebooks\\train_ambulance_colab.ipynb on Colab\n"
    "4. Put the resulting file at models\\ambulance_detector.pt\n"
    "No pretrained fallback exists (COCO has no ambulance class)."
)


class AmbulanceDetectorUnavailableError(RuntimeError):
    """Raised when the ambulance model is missing or unusable."""


class AmbulanceDetector:
    def __init__(self, weights: Optional[Path] = None, conf: float = 0.5, iou: float = 0.5,
                 imgsz: int = 640, device: str = "cpu") -> None:
        if not 0.05 <= conf <= 0.99:
            raise ValueError("conf must be within 0.05..0.99")
        if not 0.05 <= iou <= 0.95:
            raise ValueError("iou must be within 0.05..0.95")
        if not 160 <= imgsz <= 1920:
            raise ValueError("imgsz must be within 160..1920")
        self.weights = Path(weights) if weights else AMBULANCE_WEIGHTS
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self._model = None
        self._ids: set = set()

    def available(self) -> bool:
        return self.weights.exists()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.weights.exists():
            raise AmbulanceDetectorUnavailableError(HELP.format(path=self.weights))
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.weights))
        except Exception as exc:
            raise AmbulanceDetectorUnavailableError(f"Cannot load {self.weights}: {exc}") from exc
        ids = {int(i) for i, n in model.names.items()
               if any(k in str(n).lower() for k in AMBULANCE_NAME_KEYWORDS)}
        if not ids:
            raise AmbulanceDetectorUnavailableError(
                f"No class containing {AMBULANCE_NAME_KEYWORDS} in {list(model.names.values())}. "
                "Edit AMBULANCE_NAME_KEYWORDS in config/emergency_config.py.")
        self._model, self._ids = model, ids
        log.info("Ambulance detector loaded: %s classes=%s", self.weights, ids)

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            raise ValueError("predict() expects a BGR image array of shape (H, W, 3)")
        self.load()
        res = self._model.predict(frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                  device=self.device, classes=sorted(self._ids), verbose=False)[0]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy().astype(int)):
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c), "ambulance",
                                 str(self._model.names[int(k)]), int(k)))
        return out

    def model_info(self) -> dict:
        self.load()
        return {"weights": str(self.weights), "model_classes": list(self._model.names.values()),
                "conf": self.conf, "iou": self.iou, "imgsz": self.imgsz, "limitations": list(LIMITATIONS)}


@dataclass(frozen=True)
class AmbulanceSighting:
    detection: Detection
    approach: Optional[str]      # None when outside all calibrated regions
    lane_index: Optional[int]
    calibrated: bool

    @property
    def confidence(self) -> float:
        return self.detection.confidence


def locate_sightings(detections: Sequence[Detection], frame_shape: Tuple[int, ...],
                     analyzer: LaneAnalyzer) -> List[AmbulanceSighting]:
    """Map detections to approach/lane using the user-defined regions (never guessed)."""
    h, w = frame_shape[:2]
    analysis = analyzer.analyze(list(detections), frame_shape, stationary=None)
    out: List[AmbulanceSighting] = []
    for d, ap in zip(detections, analysis.assignments):
        lane = None
        if ap is not None:
            for li, poly in analyzer.config.approaches[ap].lanes.items():
                if analyzer._inside(analyzer._px(poly, w, h), d.bottom_center):
                    lane = li
                    break
        out.append(AmbulanceSighting(d, ap, lane, analysis.calibrated))
    return out


class AmbulanceConfirmer:
    """Requires `confirm_frames` detections within the last `window` frames per approach."""

    def __init__(self, min_conf: float = 0.5, confirm_frames: int = 3, window: int = 5) -> None:
        if not 1 <= confirm_frames <= window:
            raise ValueError("need 1 <= confirm_frames <= window")
        self.min_conf, self.confirm_frames, self.window = min_conf, confirm_frames, window
        self._hist: Dict[str, Deque[bool]] = {}

    def reset(self) -> None:
        self._hist.clear()

    def update(self, sightings: Sequence[AmbulanceSighting]) -> List[AmbulanceSighting]:
        best: Dict[str, AmbulanceSighting] = {}
        for s in sightings:
            if s.approach is None or s.confidence < self.min_conf:
                continue   # cannot request priority for an unknown approach
            if s.approach not in best or s.confidence > best[s.approach].confidence:
                best[s.approach] = s
        for key in set(self._hist) | set(best):
            self._hist.setdefault(key, deque(maxlen=self.window)).append(key in best)
        return [best[k] for k in best if sum(self._hist[k]) >= self.confirm_frames]