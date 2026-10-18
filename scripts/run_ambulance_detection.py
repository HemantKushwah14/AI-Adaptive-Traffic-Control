"""Run the ambulance detector on an image or video (needs models\\ambulance_detector.pt).

    python scripts\\run_ambulance_detection.py --image frame.jpg
    python scripts\\run_ambulance_detection.py --video clip.mp4 --stride 3
Approach assignment uses data\\config\\lane_regions.json (calibrate with scripts\\calibrate_regions.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from computer_vision.ambulance_detector import (LIMITATIONS, AmbulanceConfirmer,  # noqa: E402
                                                AmbulanceDetector, AmbulanceDetectorUnavailableError,
                                                locate_sightings)
from computer_vision.detector import draw_detections  # noqa: E402
from computer_vision.lane_analyzer import LaneAnalyzer, load_or_default  # noqa: E402
from config.cv_config import LANE_REGION_FILE, VISION_OUT_DIR  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path)
    g.add_argument("--video", type=Path)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--regions", type=Path, default=LANE_REGION_FILE)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=900)
    p.add_argument("--confirm", type=int, default=3, help="frames (of last 5) required to confirm")
    a = p.parse_args()
    src = a.image or a.video
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1
    try:
        det = AmbulanceDetector(a.weights, conf=a.conf)
        det.load()
        analyzer = LaneAnalyzer(load_or_default(a.regions))
        VISION_OUT_DIR.mkdir(parents=True, exist_ok=True)
        print("Limitations:\n  - " + "\n  - ".join(LIMITATIONS))
        if not analyzer.config.calibrated:
            print("WARNING: regions UNCALIBRATED - approach assignment is not meaningful.")
        if a.image:
            img = cv2.imread(str(a.image))
            if img is None:
                print("ERROR: unreadable image")
                return 1
            dets = det.predict(img)
            sightings = locate_sightings(dets, img.shape, analyzer)
            print(f"\nAmbulance detections: {len(dets)}")
            for s in sightings:
                print(f"  conf {s.confidence:.2f} | approach {s.approach} | lane {s.lane_index}")
            out = VISION_OUT_DIR / f"{a.image.stem}_ambulance.jpg"
            cv2.imwrite(str(out), draw_detections(analyzer.draw_overlay(img), dets))
            print(f"Saved {out}  (single frame: NOT confirmed; use video mode for confirmation)")
            return 0
        cap = cv2.VideoCapture(str(a.video))
        if not cap.isOpened():
            print("ERROR: cannot open video")
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        confirmer, rows, read, done = AmbulanceConfirmer(a.conf, a.confirm, 5), [], 0, 0
        try:
            while done < a.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                read += 1
                if (read - 1) % a.stride:
                    continue
                done += 1
                confirmed = confirmer.update(locate_sightings(det.predict(frame), frame.shape, analyzer))
                for s in confirmed:
                    rows.append({"time_s": (read - 1) / fps, "approach": s.approach,
                                 "lane": s.lane_index, "confidence": s.confidence})
        finally:
            cap.release()
        df = pd.DataFrame(rows, columns=["time_s", "approach", "lane", "confidence"])
        out = VISION_OUT_DIR / f"{a.video.stem}_ambulance_confirmed.csv"
        df.to_csv(out, index=False)
        print(f"\nFrames processed: {done} | confirmed ambulance frames: {len(df)}")
        print(df.groupby("approach").size().to_string() if len(df) else "No confirmed ambulance.")
        print(f"Saved {out}")
    except (AmbulanceDetectorUnavailableError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())