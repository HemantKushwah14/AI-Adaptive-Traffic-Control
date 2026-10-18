"""Interactive region calibration on a camera frame (opens an OpenCV window).

    python scripts\\calibrate_regions.py --image data\\my_camera_frame.jpg

For each approach N, S, E, W: click the polygon of the incoming road area, press Enter.
Then click an optional queue zone (area just before the stop line), press Enter (or 's' to skip).
Keys: click=add point | Enter=finish | u=undo | s=skip | q=quit without saving.
Approaches your camera cannot see: press 's' on their polygon - they stay unmeasured (None).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from computer_vision.lane_analyzer import ApproachRegion, LaneRegionConfig  # noqa: E402
from config.constants import APPROACH_NAMES, APPROACHES  # noqa: E402
from config.cv_config import LANE_REGION_FILE  # noqa: E402

WIN = "Calibrate regions"
Pt = Tuple[int, int]


def collect(base, label: str) -> Optional[List[Pt]]:
    pts: List[Pt] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        canvas = base.copy()
        for i, p in enumerate(pts):
            cv2.circle(canvas, p, 4, (0, 255, 255), -1)
            if i:
                cv2.line(canvas, pts[i - 1], p, (0, 255, 255), 2)
        if len(pts) > 2:
            cv2.line(canvas, pts[-1], pts[0], (0, 160, 255), 1)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, f"{label} | click=add  Enter=done  u=undo  s=skip  q=quit", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) >= 3:
            return list(pts)
        if k == ord("u") and pts:
            pts.pop()
        elif k == ord("s"):
            return None
        elif k == ord("q"):
            raise KeyboardInterrupt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=LANE_REGION_FILE)
    args = p.parse_args()
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"ERROR: cannot read image {args.image}")
        return 1
    h, w = img.shape[:2]
    norm = lambda poly: [(round(x / w, 5), round(y / h, 5)) for x, y in poly]  # noqa: E731
    regions = {}
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        for a in APPROACHES:
            poly = collect(img, f"{APPROACH_NAMES[a]} approach area")
            if poly is None:
                print(f"{a}: skipped (will not be measured)")
                continue
            qz = collect(img, f"{APPROACH_NAMES[a]} queue zone (optional)")
            regions[a] = ApproachRegion(norm(poly), norm(qz) if qz else None)
    except KeyboardInterrupt:
        print("Aborted, nothing saved.")
        return 1
    finally:
        cv2.destroyAllWindows()
    if not regions:
        print("ERROR: no approach defined, nothing saved.")
        return 1
    cfg = LaneRegionConfig(regions, calibrated=True, note=f"Calibrated on {args.image.name}")
    cfg.save(args.out)
    print(f"Saved {args.out} with approaches: {sorted(regions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())