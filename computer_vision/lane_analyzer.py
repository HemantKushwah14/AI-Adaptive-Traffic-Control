"""Configurable region-of-interest mapping: detection -> approach (N/S/E/W) -> lane.

IMPORTANT: which approach a vehicle belongs to depends on YOUR camera geometry. Regions are
stored as polygons in NORMALISED image coordinates (0..1) in data/config/lane_regions.json.
Create them for your footage with:  python scripts\\calibrate_regions.py --image <frame.jpg>
If the file is missing, an UNCALIBRATED wedge template is used and `calibrated=False` is
propagated to every result so the UI can warn the user.

Several approaches may be omitted (one camera often sees one approach only); omitted
approaches report density/queue = None, never a made-up number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from computer_vision.detector import Detection
from config.constants import APPROACHES, DENSITY_CAPACITY_VEH, NUM_LANES, QUEUE_NORM_VEH
from config.cv_config import CLASS_COLORS_BGR, DEFAULT_COLOR_BGR, LANE_REGION_FILE, UNIFIED_CLASSES
from utils.logging_utils import get_logger

log = get_logger(__name__)
Poly = List[Tuple[float, float]]
APPROACH_COLORS = {"N": (255, 200, 0), "S": (0, 200, 255), "E": (80, 220, 80), "W": (200, 120, 255)}


@dataclass
class ApproachRegion:
    polygon: Poly
    queue_zone: Optional[Poly] = None
    lanes: Dict[int, Poly] = field(default_factory=dict)


@dataclass
class LaneRegionConfig:
    approaches: Dict[str, ApproachRegion]
    calibrated: bool = False
    note: str = ""

    def validate(self) -> None:
        if not self.approaches:
            raise ValueError("Region config defines no approach.")

        def check(poly: Poly, what: str) -> None:
            if len(poly) < 3:
                raise ValueError(f"{what}: polygon needs >= 3 points")
            for x, y in poly:
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError(f"{what}: coordinates must be normalised to 0..1, got ({x}, {y})")

        for a, r in self.approaches.items():
            if a not in APPROACHES:
                raise ValueError(f"Unknown approach '{a}' (expected one of {APPROACHES})")
            check(r.polygon, f"{a}.polygon")
            if r.queue_zone:
                check(r.queue_zone, f"{a}.queue_zone")
            for li, poly in r.lanes.items():
                if not 0 <= li < NUM_LANES:
                    raise ValueError(f"{a}: lane index {li} outside 0..{NUM_LANES - 1}")
                check(poly, f"{a}.lane{li}")

    def to_dict(self) -> dict:
        return {"calibrated": self.calibrated, "note": self.note, "approaches": {
            a: {"polygon": r.polygon, "queue_zone": r.queue_zone,
                "lanes": {str(k): v for k, v in r.lanes.items()}} for a, r in self.approaches.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "LaneRegionConfig":
        regs = {}
        for a, r in d.get("approaches", {}).items():
            regs[a] = ApproachRegion(
                polygon=[tuple(p) for p in r["polygon"]],
                queue_zone=[tuple(p) for p in r["queue_zone"]] if r.get("queue_zone") else None,
                lanes={int(k): [tuple(p) for p in v] for k, v in (r.get("lanes") or {}).items()})
        cfg = cls(regs, bool(d.get("calibrated", False)), d.get("note", ""))
        cfg.validate()
        return cfg

    def save(self, path: Path = LANE_REGION_FILE) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path = LANE_REGION_FILE) -> "LaneRegionConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid region file {path}: {exc}") from exc

    @classmethod
    def default_template(cls) -> "LaneRegionConfig":
        c = (0.5, 0.5)
        return cls({
            "N": ApproachRegion([(0, 0), (1, 0), c], [(0.3, 0.2), (0.7, 0.2), c]),
            "S": ApproachRegion([(0, 1), (1, 1), c], [(0.3, 0.8), (0.7, 0.8), c]),
            "E": ApproachRegion([(1, 0), (1, 1), c], [(0.8, 0.3), (0.8, 0.7), c]),
            "W": ApproachRegion([(0, 0), (0, 1), c], [(0.2, 0.3), (0.2, 0.7), c]),
        }, calibrated=False,
            note="UNCALIBRATED demo wedges. Results do not reflect real approach geometry.")


def load_or_default(path: Path = LANE_REGION_FILE) -> LaneRegionConfig:
    if Path(path).exists():
        return LaneRegionConfig.load(path)
    log.warning("No %s found - using UNCALIBRATED wedge template.", path)
    return LaneRegionConfig.default_template()


@dataclass
class TrafficAnalysis:
    counts: Dict[str, Optional[int]]
    class_counts: Dict[str, Dict[str, int]]
    lane_counts: Dict[str, Dict[int, int]]
    density: Dict[str, Optional[float]]
    queue: Dict[str, Optional[int]]
    queue_method: str
    unassigned: int
    calibrated: bool
    assignments: List[Optional[str]]

    def density_vector(self) -> List[float]:
        """Density in APPROACHES order (None -> 0.0); matches DQN state slots [0:4]."""
        return [float(self.density[a] or 0.0) for a in APPROACHES]

    def queue_vector(self) -> List[float]:
        """Normalised queue in APPROACHES order (None -> 0.0); matches DQN state slots [4:8]."""
        return [min(float(self.queue[a] or 0) / QUEUE_NORM_VEH, 1.0) for a in APPROACHES]

    def to_dict(self) -> dict:
        return {"counts": self.counts, "class_counts": self.class_counts,
                "lane_counts": {a: {str(k): v for k, v in d.items()} for a, d in self.lane_counts.items()},
                "density": self.density, "queue": self.queue, "queue_method": self.queue_method,
                "unassigned": self.unassigned, "calibrated": self.calibrated}


class LaneAnalyzer:
    def __init__(self, config: Optional[LaneRegionConfig] = None,
                 capacity: float = DENSITY_CAPACITY_VEH) -> None:
        self.config = config or load_or_default()
        self.config.validate()
        self.capacity = float(capacity)

    @staticmethod
    def _px(poly: Poly, w: int, h: int) -> np.ndarray:
        return (np.asarray(poly, dtype=np.float32) * np.array([w, h], dtype=np.float32)).reshape(-1, 1, 2)

    @staticmethod
    def _inside(contour: np.ndarray, pt: Tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0

    def analyze(self, detections: Sequence[Detection], frame_shape: Tuple[int, ...],
                stationary: Optional[Sequence[bool]] = None) -> TrafficAnalysis:
        h, w = frame_shape[:2]
        regs = {a: (self._px(r.polygon, w, h),
                    self._px(r.queue_zone, w, h) if r.queue_zone else None,
                    {li: self._px(p, w, h) for li, p in r.lanes.items()})
                for a, r in self.config.approaches.items()}
        counts: Dict[str, Optional[int]] = {a: (0 if a in regs else None) for a in APPROACHES}
        cls_counts = {a: {c: 0 for c in UNIFIED_CLASSES} for a in APPROACHES}
        lane_counts = {a: {i: 0 for i in range(NUM_LANES)} for a in APPROACHES}
        queue: Dict[str, Optional[int]] = {a: None for a in APPROACHES}
        if stationary is not None:
            for a in regs:
                queue[a] = 0
        assign: List[Optional[str]] = []
        unassigned = 0
        for i, d in enumerate(detections):
            pt = d.bottom_center
            owner = None
            for a, (poly, qz, lanes) in regs.items():
                if self._inside(poly, pt):
                    owner = a
                    counts[a] += 1                                   # type: ignore[operator]
                    cls_counts[a][d.label] = cls_counts[a].get(d.label, 0) + 1
                    for li, lp in lanes.items():
                        if self._inside(lp, pt):
                            lane_counts[a][li] += 1
                            break
                    if stationary is not None and stationary[i] and (qz is None or self._inside(qz, pt)):
                        queue[a] += 1                                # type: ignore[operator]
                    break
            assign.append(owner)
            unassigned += int(owner is None)
        density = {a: (min(counts[a] / self.capacity, 1.0) if counts[a] is not None else None)  # type: ignore
                   for a in APPROACHES}
        method = ("stationary tracked vehicles inside queue zone (heuristic)" if stationary is not None
                  else "unavailable from a single frame (needs video tracking)")
        return TrafficAnalysis(counts, cls_counts, lane_counts, density, queue, method, unassigned,
                               self.config.calibrated, assign)

    def draw_overlay(self, frame: np.ndarray, analysis: Optional[TrafficAnalysis] = None) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()
        layer = frame.copy()
        for a, r in self.config.approaches.items():
            col = APPROACH_COLORS[a]
            cv2.fillPoly(layer, [self._px(r.polygon, w, h).astype(np.int32)], col)
        out = cv2.addWeighted(layer, 0.18, out, 0.82, 0)
        for a, r in self.config.approaches.items():
            col = APPROACH_COLORS[a]
            poly = self._px(r.polygon, w, h).astype(np.int32)
            cv2.polylines(out, [poly], True, col, 2)
            if r.queue_zone:
                cv2.polylines(out, [self._px(r.queue_zone, w, h).astype(np.int32)], True, col, 1, cv2.LINE_AA)
            anchor = tuple(int(v) for v in poly.reshape(-1, 2).mean(axis=0))
            label = a if analysis is None else f"{a}: {analysis.counts[a]}"
            cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 1, cv2.LINE_AA)
        if not self.config.calibrated:
            cv2.putText(out, "UNCALIBRATED REGIONS", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2, cv2.LINE_AA)
        return out