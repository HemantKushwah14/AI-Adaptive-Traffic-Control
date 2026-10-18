"""Turn a downloaded ambulance dataset (YOLO format, e.g. a Roboflow export) into a clean
SINGLE-CLASS dataset: data/ambulance_yolo.

Handles: source train/valid/test folders or flat folders, polygon labels (-> boxes), content-hash
de-duplication across splits (no exact-duplicate leakage), optional hard negatives from your
DETRAC YOLO dataset (vehicles that are NOT ambulances).

    python -m dataset.prepare_ambulance_dataset --list-classes
    python -m dataset.prepare_ambulance_dataset --ambulance-classes ambulance --overwrite
    python -m dataset.prepare_ambulance_dataset --negatives-from data\\detrac_yolo --max-negatives 400 --overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from config.cv_config import YOLO_DIR
from config.emergency_config import AMBULANCE_RAW_DIR, AMBULANCE_YOLO_DIR
from utils.logging_utils import get_logger

log = get_logger(__name__)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ALIASES = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
ORDER = {"train": 0, "val": 1, "test": 2}


def load_names(src: Path) -> Dict[int, str]:
    for n in ("data.yaml", "dataset.yaml"):
        p = src / n
        if p.exists():
            names = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("names")
            if isinstance(names, list):
                return {i: str(x) for i, x in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
    return {}


def label_for(img: Path) -> Optional[Path]:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
            break
    side = img.with_suffix(".txt")
    return side if side.exists() else None


def split_of(img: Path, src: Path) -> Optional[str]:
    for part in img.relative_to(src).parts[:-1]:
        if part.lower() in ALIASES:
            return ALIASES[part.lower()]
    return None


def parse_label(path: Path, keep_ids: Set[int]) -> Tuple[List[str], int]:
    """Return (single-class YOLO lines, malformed count). Polygons become bounding boxes."""
    lines, bad = [], 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
            if cid not in keep_ids:
                continue
            if len(vals) == 4:
                cx, cy, w, h = vals
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                xs, ys = vals[0::2], vals[1::2]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            else:
                raise ValueError("unexpected field count")
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                raise ValueError("degenerate box")
            lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        except ValueError:
            bad += 1
    return lines, bad


def list_classes(src: Path) -> int:
    if not src.exists():
        print(f"[FAIL] {src} not found. Download a YOLO-format ambulance dataset into it.")
        return 1
    names = load_names(src)
    print("Classes declared in data.yaml:", names or "(no data.yaml found)")
    cnt: Counter = Counter()
    n_img = 0
    for img in src.rglob("*"):
        if img.suffix.lower() in IMG_EXT:
            n_img += 1
            lbl = label_for(img)
            if lbl:
                for ln in lbl.read_text(encoding="utf-8").split("\n"):
                    if ln.split():
                        cnt[int(float(ln.split()[0]))] += 1
    print(f"Images: {n_img}")
    for cid, n in sorted(cnt.items()):
        print(f"  class {cid} ({names.get(cid, '?')}): {n} boxes")
    return 0


def sample_negatives(neg_dir: Path, max_n: int, seed: int) -> Dict[str, List[Path]]:
    rng = random.Random(seed)
    quota = {"train": round(max_n * 0.70), "val": round(max_n * 0.15)}
    quota["test"] = max_n - quota["train"] - quota["val"]
    out: Dict[str, List[Path]] = {}
    for sp, q in quota.items():
        d = neg_dir / "images" / sp
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []
        rng.shuffle(imgs)
        out[sp] = imgs[:q]
    return out


def prepare(src: Path, out: Path, keep_names: Set[str], keep_ids_opt: Optional[Set[int]],
            negatives_from: Optional[Path], max_negatives: int, seed: int, overwrite: bool) -> dict:
    if not src.exists():
        raise FileNotFoundError(f"{src} not found.")
    id2name = load_names(src)
    if keep_ids_opt:
        keep_ids = set(keep_ids_opt)
    else:
        if not id2name:
            raise ValueError("No data.yaml in the source; pass --ambulance-ids 0 (class ids to keep).")
        keep_ids = {i for i, n in id2name.items() if n.strip().lower() in keep_names}
        if not keep_ids:
            raise ValueError(f"None of {sorted(keep_names)} appear in source classes {id2name}. "
                             "Use --ambulance-classes or --ambulance-ids.")
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        raise ValueError(f"No images under {src}")

    skipped_unlabelled, recs = 0, []
    for img in images:
        lbl = label_for(img)
        if lbl is None:
            skipped_unlabelled += 1
            continue
        dg = hashlib.md5(img.read_bytes()).hexdigest()
        sp = split_of(img, src)
        if sp is None:
            b = int(dg[:8], 16) % 100
            sp = "train" if b < 70 else "val" if b < 85 else "test"
        recs.append((sp, img, lbl, dg))
    had_source_splits = any(split_of(r[1], src) for r in recs)

    recs.sort(key=lambda r: (ORDER[r[0]], str(r[1])))
    seen: Dict[str, str] = {}
    kept, dup_dropped = [], 0
    for sp, img, lbl, dg in recs:
        if dg in seen:
            dup_dropped += 1
            continue
        seen[dg] = sp
        kept.append((sp, img, lbl, dg))

    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists. Use --overwrite.")
        shutil.rmtree(out)
    for sp in ORDER:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    stats = {sp: {"images": 0, "positive_images": 0, "negative_images": 0, "boxes": 0} for sp in ORDER}
    bad_total = 0
    for sp, img, lbl, dg in kept:
        lines, bad = parse_label(lbl, keep_ids)
        bad_total += bad
        stem = f"{img.stem}_{dg[:8]}"
        shutil.copy2(img, out / "images" / sp / f"{stem}{img.suffix.lower()}")
        (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[sp]["images"] += 1
        stats[sp]["boxes"] += len(lines)
        stats[sp]["positive_images" if lines else "negative_images"] += 1

    neg_added = 0
    if negatives_from:
        if max_negatives < 1:
            raise ValueError("--max-negatives must be >= 1")
        for sp, imgs in sample_negatives(Path(negatives_from), max_negatives, seed).items():
            for img in imgs:
                stem = f"neg_{img.stem}"
                shutil.copy2(img, out / "images" / sp / f"{stem}{img.suffix.lower()}")
                (out / "labels" / sp / f"{stem}.txt").write_text("")
                stats[sp]["images"] += 1
                stats[sp]["negative_images"] += 1
                neg_added += 1

    for sp, st in stats.items():
        if st["images"] == 0 or st["boxes"] == 0:
            raise ValueError(f"Split '{sp}' has {st['images']} images and {st['boxes']} ambulance boxes. "
                             "The source needs ambulances in train, val AND test.")

    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
         "test": "images/test", "names": {0: "ambulance"}}, sort_keys=False))
    warnings = []
    if not had_source_splits:
        warnings.append("Source had no train/val/test folders; split by content hash. Near-duplicate "
                        "frames (video, augmentation) can still leak between splits.")
    if dup_dropped:
        warnings.append(f"{dup_dropped} exact-duplicate images were dropped (evaluation splits kept clean).")
    if skipped_unlabelled:
        warnings.append(f"{skipped_unlabelled} images without a label file were skipped.")
    report = {"source_classes": id2name, "kept_source_class_ids": sorted(keep_ids), "splits": stats,
              "negatives_added": neg_added, "malformed_label_lines": bad_total, "warnings": warnings,
              "seed": seed}
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare a single-class ambulance YOLO dataset.")
    p.add_argument("--src", type=Path, default=AMBULANCE_RAW_DIR)
    p.add_argument("--out", type=Path, default=AMBULANCE_YOLO_DIR)
    p.add_argument("--list-classes", action="store_true")
    p.add_argument("--ambulance-classes", default="ambulance", help="comma-separated source class names")
    p.add_argument("--ambulance-ids", default=None, help="comma-separated source class ids (overrides names)")
    p.add_argument("--negatives-from", type=Path, default=None,
                   help=f"YOLO dataset providing non-ambulance images, e.g. {YOLO_DIR}")
    p.add_argument("--max-negatives", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.list_classes:
        return list_classes(a.src)
    try:
        ids = {int(x) for x in a.ambulance_ids.split(",")} if a.ambulance_ids else None
        names = {x.strip().lower() for x in a.ambulance_classes.split(",") if x.strip()}
        rep = prepare(a.src, a.out, names, ids, a.negatives_from, a.max_negatives, a.seed, a.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    for sp, st in rep["splits"].items():
        print(f"{sp:5s}: {st}")
    for w in rep["warnings"]:
        print("WARNING:", w)
    print(f"Wrote {a.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())