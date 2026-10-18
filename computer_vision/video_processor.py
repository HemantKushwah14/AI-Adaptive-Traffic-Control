"""TrafficVisionPipeline: detector + tracker + lane analysis for images and videos."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from computer_vision.detector import Detection, VehicleDetector, draw_detections
from computer_vision.lane_analyzer import LaneAnalyzer, TrafficAnalysis
from computer_vision.vehicle_counter import VehicleCounter, count_by_class
from config.constants import APPROACHES
from utils.logging_utils import get_logger

log = get_logger(__name__)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes (e.g. from Streamlit) into a BGR array."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The file is not a readable image (jpg/png expected).")
    return img


def image_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


@dataclass
class ImageResult:
    annotated: np.ndarray
    detections: List[Detection]
    class_counts: Dict[str, int]
    analysis: TrafficAnalysis
    inference_ms: float


@dataclass
class VideoResult:
    output_path: Path
    codec: str
    frames_read: int
    frames_processed: int
    source_fps: float
    summary: Dict
    per_frame: pd.DataFrame
    sample_frames: List[np.ndarray] = field(default_factory=list)


class TrafficVisionPipeline:
    def __init__(self, detector: Optional[VehicleDetector] = None,
                 analyzer: Optional[LaneAnalyzer] = None) -> None:
        self.detector = detector or VehicleDetector()
        self.analyzer = analyzer or LaneAnalyzer()

    # ---------- single image ----------
    def process_image(self, bgr: np.ndarray) -> ImageResult:
        t0 = time.perf_counter()
        dets = self.detector.predict(bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        analysis = self.analyzer.analyze(dets, bgr.shape, stationary=None)
        canvas = self.analyzer.draw_overlay(bgr, analysis)
        canvas = draw_detections(canvas, dets)
        return ImageResult(canvas, dets, count_by_class(dets), analysis, ms)

    # ---------- video ----------
    @staticmethod
    def _open_writer(path: Path, fps: float, size: Tuple[int, int]):
        for codec in ("avc1", "mp4v"):
            wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
            if wr.isOpened():
                return wr, codec
            wr.release()
        raise RuntimeError(f"Could not open a video writer for {path} (tried avc1, mp4v).")

    def process_video(self, input_path: Path, output_path: Path, stride: int = 3,
                      max_frames: int = 1500,
                      progress_cb: Optional[Callable[[int, int], None]] = None) -> VideoResult:
        if stride < 1 or max_frames < 1:
            raise ValueError("stride and max_frames must be >= 1")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        writer = None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                raise ValueError("Video has invalid dimensions.")
            target = min(max_frames, math.ceil(total / stride)) if total > 0 else max_frames
            counter = VehicleCounter(max_distance=0.09 * w, max_missed=8, min_hits=3,
                                     min_points=10, stationary_px=0.008 * w)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer, codec = self._open_writer(output_path, max(fps / stride, 1.0), (w, h))

            rows: List[dict] = []
            samples: List[np.ndarray] = []
            infer_ms: List[float] = []
            read = processed = 0
            sample_every = max(1, target // 6)
            while processed < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                read += 1
                if (read - 1) % stride:
                    continue
                t0 = time.perf_counter()
                dets = self.detector.predict(frame)
                infer_ms.append((time.perf_counter() - t0) * 1000.0)
                # two-pass: assign approaches first (without queue), then track, then re-analyse
                pre = self.analyzer.analyze(dets, frame.shape, stationary=None)
                tracking = counter.update(dets, pre.assignments)
                analysis = self.analyzer.analyze(dets, frame.shape, stationary=tracking.stationary)
                canvas = self.analyzer.draw_overlay(frame, analysis)
                canvas = draw_detections(canvas, dets)
                for i, d in enumerate(dets):
                    if tracking.stationary[i]:
                        cx, cy = map(int, d.center)
                        cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)
                writer.write(canvas)
                row = {"frame": read - 1, "time_s": (read - 1) / fps, "total": len(dets),
                       "inference_ms": infer_ms[-1]}
                for a in APPROACHES:
                    row[f"count_{a}"] = analysis.counts[a]
                    row[f"queue_{a}"] = analysis.queue[a]
                row.update({f"cls_{k}": v for k, v in tracking.class_counts.items()})
                rows.append(row)
                if processed % sample_every == 0 and len(samples) < 6:
                    samples.append(canvas)
                processed += 1
                if progress_cb:
                    progress_cb(processed, target)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if processed == 0:
            raise ValueError("No frames could be read from the video.")

        df = pd.DataFrame(rows)
        summary = {
            "frames_processed": processed,
            "avg_inference_ms": float(np.mean(infer_ms)),
            "unique_vehicles_total": counter.unique_total,
            "unique_vehicles_by_class": dict(counter.unique_by_class),
            "unique_vehicles_by_approach": {a: dict(c) for a, c in counter.unique_by_approach.items()},
            "avg_count_by_approach": {a: (float(df[f"count_{a}"].mean()) if df[f"count_{a}"].notna().any() else None)
                                      for a in APPROACHES},
            "peak_count_by_approach": {a: (int(df[f"count_{a}"].max()) if df[f"count_{a}"].notna().any() else None)
                                       for a in APPROACHES},
            "avg_queue_by_approach": {a: (float(df[f"queue_{a}"].mean()) if df[f"queue_{a}"].notna().any() else None)
                                      for a in APPROACHES},
            "regions_calibrated": self.analyzer.config.calibrated,
            "model": self.detector.model_info(),
            "tracking_note": "Unique counts come from a simple centroid tracker; ID switches can "
                             "over/under-count. Treat as estimates.",
        }
        return VideoResult(Path(output_path), codec, read, processed, float(fps), summary, df, samples)