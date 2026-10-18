"""Phase 3 milestone: run real detection + counting + lane analysis from the command line.

    python scripts\\run_detection.py --image path\\to\\frame.jpg
    python scripts\\run_detection.py --video path\\to\\clip.mp4 --stride 3
    python scripts\\run_detection.py --image frame.jpg --weights models\\vehicle_detector.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from computer_vision.detector import DetectorUnavailableError, VehicleDetector  # noqa: E402
from computer_vision.lane_analyzer import LaneAnalyzer, LaneRegionConfig, load_or_default  # noqa: E402
from computer_vision.video_processor import TrafficVisionPipeline  # noqa: E402
from config.cv_config import LANE_REGION_FILE, VISION_OUT_DIR  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path)
    g.add_argument("--video", type=Path)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--regions", type=Path, default=LANE_REGION_FILE)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=600)
    a = p.parse_args()

    src = a.image or a.video
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1
    try:
        detector = VehicleDetector(weights=a.weights, conf=a.conf)
        pipe = TrafficVisionPipeline(detector, LaneAnalyzer(load_or_default(a.regions)))
        info = detector.model_info()
        print(f"Model mode: {info['mode']} | weights: {info['weights']}\nNote: {info['note']}")
        if not pipe.analyzer.config.calibrated:
            print("WARNING: lane regions are UNCALIBRATED; per-approach numbers are not meaningful. "
                  "Run scripts\\calibrate_regions.py")
        VISION_OUT_DIR.mkdir(parents=True, exist_ok=True)
        if a.image:
            img = cv2.imread(str(a.image))
            if img is None:
                print("ERROR: not a readable image")
                return 1
            r = pipe.process_image(img)
            out = VISION_OUT_DIR / f"{a.image.stem}_detected.jpg"
            cv2.imwrite(str(out), r.annotated)
            print(f"\nDetections: {len(r.detections)}  | inference {r.inference_ms:.0f} ms")
            print("Class counts:", r.class_counts)
            print("Per-approach counts:", r.analysis.counts)
            print("Density (0..1):", {k: (round(v, 3) if v is not None else None) for k, v in r.analysis.density.items()})
            print("Queue:", r.analysis.queue, "-", r.analysis.queue_method)
            print(f"Saved {out}")
        else:
            out = VISION_OUT_DIR / f"{a.video.stem}_detected.mp4"
            r = pipe.process_video(a.video, out, a.stride, a.max_frames,
                                   progress_cb=lambda d, t: print(f"\rframes {d}/{t}", end=""))
            print()
            (VISION_OUT_DIR / f"{a.video.stem}_summary.json").write_text(json.dumps(r.summary, indent=2, default=str))
            r.per_frame.to_csv(VISION_OUT_DIR / f"{a.video.stem}_per_frame.csv", index=False)
            print(json.dumps({k: v for k, v in r.summary.items() if k != "model"}, indent=2, default=str))
            print(f"Saved {r.output_path} (codec {r.codec})")
    except (DetectorUnavailableError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())