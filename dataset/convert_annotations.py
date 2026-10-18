"""UA-DETRAC XML parsing and YOLO label conversion (pure functions, no file copying).

UA-DETRAC XML layout:
  <sequence name="MVI_20011">
    <ignored_region><box left= top= width= height=/></ignored_region>
    <frame num="1" density="..">
      <target_list>
        <target id="1"><box left= top= width= height=/>
                       <attribute vehicle_type="car" .../></target>
Box coordinates are absolute pixels (top-left origin). YOLO wants normalised cx, cy, w, h.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config.cv_config import DETRAC_PREFERRED_ORDER


@dataclass(frozen=True)
class Box:
    cls: str
    left: float
    top: float
    width: float
    height: float


@dataclass
class SequenceAnnotation:
    name: str
    frames: Dict[int, List[Box]] = field(default_factory=dict)
    ignored_regions: List[Tuple[float, float, float, float]] = field(default_factory=list)
    malformed: int = 0


def _f(el: ET.Element, key: str) -> float:
    return float(el.get(key))  # type: ignore[arg-type]  (None -> TypeError, caught by callers)


def parse_detrac_xml(path: Path) -> SequenceAnnotation:
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"{path}: cannot parse XML ({exc})") from exc
    if root.tag != "sequence":
        raise ValueError(f"{path}: root tag is '{root.tag}', expected 'sequence'")
    ann = SequenceAnnotation(name=root.get("name") or path.stem)

    ign = root.find("ignored_region")
    if ign is not None:
        for b in ign.findall("box"):
            try:
                ann.ignored_regions.append((_f(b, "left"), _f(b, "top"), _f(b, "width"), _f(b, "height")))
            except (TypeError, ValueError):
                ann.malformed += 1

    for fr in root.findall("frame"):
        try:
            num = int(fr.get("num"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            ann.malformed += 1
            continue
        boxes: List[Box] = []
        tl = fr.find("target_list")
        for t in (tl.findall("target") if tl is not None else []):
            b, a = t.find("box"), t.find("attribute")
            if b is None or a is None:
                ann.malformed += 1
                continue
            try:
                vt = (a.get("vehicle_type") or "").strip().lower()
                if not vt:
                    raise ValueError("empty vehicle_type")
                boxes.append(Box(vt, _f(b, "left"), _f(b, "top"), _f(b, "width"), _f(b, "height")))
            except (TypeError, ValueError):
                ann.malformed += 1
        ann.frames[num] = boxes
    return ann


def box_to_yolo(box: Box, img_w: int, img_h: int, min_px: float = 2.0
                ) -> Optional[Tuple[float, float, float, float]]:
    """Clip to the image and normalise. Returns None for degenerate boxes."""
    x1, y1 = max(0.0, box.left), max(0.0, box.top)
    x2, y2 = min(float(img_w), box.left + box.width), min(float(img_h), box.top + box.height)
    if x2 - x1 < min_px or y2 - y1 < min_px:
        return None
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h)


def parse_merge_args(items: Optional[Sequence[str]]) -> Dict[str, str]:
    """['van=car', 'others=truck'] -> {'van': 'car', 'others': 'truck'}"""
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--merge expects SRC=DST (e.g. van=car), got '{it}'")
        src, dst = (s.strip().lower() for s in it.split("=", 1))
        if not src or not dst:
            raise ValueError(f"--merge expects SRC=DST, got '{it}'")
        out[src] = dst
    return out


def build_class_list(found: Iterable[str], merges: Dict[str, str]) -> Tuple[List[str], Dict[str, int]]:
    """Training class list is built from the classes ACTUALLY present (no invented labels)."""
    names = sorted(set(found), key=lambda n: (DETRAC_PREFERRED_ORDER.index(n)
                                              if n in DETRAC_PREFERRED_ORDER else 99, n))
    targets: List[str] = []
    for n in names:
        t = merges.get(n, n)
        if t not in targets:
            targets.append(t)
    mapping = {n: targets.index(merges.get(n, n)) for n in names}
    return targets, mapping


def boxes_to_label_lines(boxes: Sequence[Box], mapping: Dict[str, int],
                         img_w: int, img_h: int) -> Tuple[List[str], int]:
    """Return (YOLO label lines, number of dropped degenerate/unmapped boxes)."""
    lines, dropped = [], 0
    for b in boxes:
        cid = mapping.get(b.cls)
        yolo = box_to_yolo(b, img_w, img_h)
        if cid is None or yolo is None:
            dropped += 1
            continue
        lines.append(f"{cid} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")
    return lines, dropped