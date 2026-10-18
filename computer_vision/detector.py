"""YOLO vehicle detector (CPU-friendly) with honest model-mode reporting.

Modes
-----
finetuned       : models/vehicle_detector.pt exists (YOU trained it on Colab).
pretrained-coco : COCO yolov8n.pt (car, motorcycle, bus, truck) - NOT fine-tuned on DETRAC.
custom          : explicit weights path.
No fake predictions are ever produced: if no model can be loaded an error is raised.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config.cv_config import (CLASS_COLORS_BGR, DEFAULT_COLOR_BGR, MODEL_NAME_TO_UNIFIED,
                              PRETRAINED_NAME, PRETRAINED_WEIGHTS, UNIFIED_CLASSES, VEHICLE_WEIGHTS)
from config.settings import MODELS_DIR
from utils.logging_utils import get_logger

log = get_logger(__name__)


class DetectorUnavailableError(RuntimeError):
    """Raised when no detection model can be loaded."""


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str          # unified class (car / motorcycle / bus / truck / ambulance)
    raw_label: str      # class name exactly as the model reports it
    class_id: int

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Ground-contact point; used for region assignment."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class VehicleDetector:
    def __init__(self, weights: Optional[Path] = None, conf: float = 0.25, iou: float = 0.5,
                 imgsz: int = 640, device: str = "cpu",
                 name_map: Optional[Dict[str, str]] = None) -> None:
        if not 0.01 <= conf <= 0.99:
            raise ValueError("conf must be within 0.01..0.99")
        if not 0.05 <= iou <= 0.95:
            raise ValueError("iou must be within 0.05..0.95")
        if imgsz < 160 or imgsz > 1920:
            raise ValueError("imgsz must be within 160..1920")
        self.requested_weights = Path(weights) if weights else None
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.name_map = {k.lower(): v for k, v in (name_map or MODEL_NAME_TO_UNIFIED).items()}
        self._model = None
        self._keep: Dict[int, str] = {}
        self.weights_path: str = ""
        self.mode: str = "not-loaded"

    # ---- loading ----
    def _ensure_pretrained(self) -> str:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if not PRETRAINED_WEIGHTS.exists():
            try:
                from ultralytics.utils.downloads import attempt_download_asset
                attempt_download_asset(str(PRETRAINED_WEIGHTS))
            except Exception as exc:
                log.warning("Direct download of %s failed (%s); letting ultralytics retry.",
                            PRETRAINED_NAME, exc)
        return str(PRETRAINED_WEIGHTS) if PRETRAINED_WEIGHTS.exists() else PRETRAINED_NAME

    def _resolve(self) -> Tuple[str, str]:
        if self.requested_weights is not None:
            if not self.requested_weights.exists():
                raise DetectorUnavailableError(f"Weights not found: {self.requested_weights}")
            return str(self.requested_weights), "custom"
        if VEHICLE_WEIGHTS.exists():
            return str(VEHICLE_WEIGHTS), "finetuned"
        return self._ensure_pretrained(), "pretrained-coco"

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailableError("ultralytics is not installed (pip install -r requirements.txt)") from exc
        path, mode = self._resolve()
        try:
            model = YOLO(path)
        except Exception as exc:
            raise DetectorUnavailableError(
                f"Could not load YOLO weights '{path}': {exc}\n"
                f"First run with pretrained weights needs internet to download {PRETRAINED_NAME}. "
                f"Alternatively download it manually into {MODELS_DIR}.") from exc
        keep = {int(i): self.name_map[str(n).lower()] for i, n in model.names.items()
                if str(n).lower() in self.name_map}
        if not keep:
            raise DetectorUnavailableError(
                f"None of the model classes {list(model.names.values())} map to vehicle classes. "
                "Edit MODEL_NAME_TO_UNIFIED in config/cv_config.py.")
        self._model, self._keep, self.weights_path, self.mode = model, keep, path, mode
        log.info("Detector loaded: mode=%s weights=%s classes=%s", mode, path, keep)

    # ---- inference ----
    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            raise ValueError("predict() expects a BGR image array of shape (H, W, 3)")
        self.load()
        res = self._model.predict(frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                  device=self.device, classes=sorted(self._keep), verbose=False)[0]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            if int(k) in self._keep:
                out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c),
                                     self._keep[int(k)], str(self._model.names[int(k)]), int(k)))
        return out

    def model_info(self) -> dict:
        self.load()
        notes = {
            "finetuned": "Fine-tuned on your converted UA-DETRAC data (see Colab evaluation JSON).",
            "pretrained-coco": "COCO-pretrained, NOT fine-tuned on UA-DETRAC. Train on Colab to adapt it.",
            "custom": "User-supplied weights.",
        }
        return {"mode": self.mode, "weights": self.weights_path,
                "model_classes": list(self._model.names.values()),
                "reported_classes": sorted(set(self._keep.values())),
                "conf": self.conf, "iou": self.iou, "imgsz": self.imgsz, "device": self.device,
                "note": notes.get(self.mode, "")}


# --------------------------------------------------------------------------- #
def draw_detections(frame: np.ndarray, detections: Sequence[Detection],
                    thickness: int = 2) -> np.ndarray:
    """Return a copy of `frame` with labelled boxes."""
    out = frame.copy()
    for d in detections:
        color = CLASS_COLORS_BGR.get(d.label, DEFAULT_COLOR_BGR)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, thickness)
        text = f"{d.label} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(p1[1], th + base + 2)
        cv2.rectangle(out, (p1[0], y0 - th - base - 2), (p1[0] + tw + 4, y0), color, -1)
        cv2.putText(out, text, (p1[0] + 2, y0 - base - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out