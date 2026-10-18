"""Build a YOLO-format dataset from the downloaded UA-DETRAC subset.

Steps: discover sequences (images + XML) -> parse annotations -> discover classes ->
split BY SEQUENCE (no leakage) -> write images/labels -> data.yaml + prepare_report.json

Usage:
    python -m dataset.prepare_dataset --overwrite
    python -m dataset.prepare_dataset --merge van=car --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image

from config.cv_config import RAW_DIR, SEQUENCE_REGEX, YOLO_DIR
from dataset.convert_annotations import (SequenceAnnotation, boxes_to_label_lines, build_class_list,
                                         parse_detrac_xml, parse_merge_args)
from utils.logging_utils import get_logger

log = get_logger(__name__)

IMG_EXT = (".jpg", ".jpeg", ".png")
SEQ_RE = re.compile(SEQUENCE_REGEX)


def find_sequence_dirs(raw_dir: Path) -> Dict[str, Path]:
    best: Dict[str, tuple] = {}
    for root, _dirs, files in os.walk(raw_dir):
        name = Path(root).name
        if SEQ_RE.fullmatch(name):
            n = sum(1 for f in files if f.lower().endswith(IMG_EXT))
            if n and (name not in best or n > best[name][1]):
                best[name] = (Path(root), n)
    return {k: v[0] for k, v in best.items()}


def find_annotation_files(raw_dir: Path) -> Dict[str, Path]:
    cands: Dict[str, List[Path]] = defaultdict(list)
    for p in raw_dir.rglob("*.xml"):
        m = SEQ_RE.search(p.stem)
        if m:
            cands[m.group(0)].append(p)
    out: Dict[str, Path] = {}
    for seq, paths in cands.items():
        exact = [p for p in paths if p.stem == seq]
        out[seq] = sorted(exact or paths)[0]
    return out


def index_frames(seq_dir: Path) -> Dict[int, Path]:
    frames: Dict[int, Path] = {}
    for p in seq_dir.iterdir():
        if p.suffix.lower() in IMG_EXT:
            m = re.search(r"(\d+)$", p.stem)
            if m:
                frames[int(m.group(1))] = p
    return frames


def split_sequences(names: List[str], train: float, val: float, seed: int) -> Dict[str, List[str]]:
    names = sorted(names)
    if len(names) < 3:
        raise ValueError(f"Need at least 3 annotated sequences for train/val/test, found {len(names)}.")
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("Ratios must satisfy 0 < train, val and train + val < 1.")
    random.Random(seed).shuffle(names)
    n = len(names)
    n_val = max(1, round(n * val))
    n_test = max(1, round(n * (1 - train - val)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough sequences for the requested split.")
    return {"train": names[:n_train], "val": names[n_train:n_train + n_val],
            "test": names[n_train + n_val:]}


def prepare(raw_dir: Path, out_dir: Path, merges: Dict[str, str], train_ratio: float = 0.70,
            val_ratio: float = 0.15, seed: int = 42, frame_stride: int = 1,
            overwrite: bool = False) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} not found. Run: python -m dataset.download_dataset")
    seq_dirs, xmls = find_sequence_dirs(raw_dir), find_annotation_files(raw_dir)
    usable = sorted(set(seq_dirs) & set(xmls))
    log.info("Image sequences: %d | XML files: %d | usable (both): %d",
             len(seq_dirs), len(xmls), len(usable))
    if not usable:
        raise ValueError("No sequence has both images and annotations. Run validate_dataset --raw.")

    anns: Dict[str, SequenceAnnotation] = {}
    for s in usable:
        try:
            anns[s] = parse_detrac_xml(xmls[s])
        except ValueError as exc:
            log.warning("Skipping %s: %s", s, exc)
    found: Counter = Counter()
    for a in anns.values():
        for boxes in a.frames.values():
            found.update(b.cls for b in boxes)
    if not found:
        raise ValueError("Annotations contain no boxes.")
    class_names, mapping = build_class_list(found, merges)
    log.info("Classes found in annotations: %s", dict(found))
    log.info("Training classes: %s | mapping: %s", class_names, mapping)

    splits = split_sequences(list(anns), train_ratio, val_ratio, seed)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists. Use --overwrite to rebuild.")
        shutil.rmtree(out_dir)

    stats = {sp: {"images": 0, "instances": Counter(), "empty_images": 0} for sp in splits}
    dropped_total = 0
    for sp, seqs in splits.items():
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            frames = index_frames(seq_dirs[seq])
            if not frames:
                continue
            with Image.open(next(iter(frames.values()))) as im:
                w, h = im.size
            for num in sorted(set(frames) & set(anns[seq].frames)):
                if frame_stride > 1 and (num - 1) % frame_stride:
                    continue
                lines, dropped = boxes_to_label_lines(anns[seq].frames[num], mapping, w, h)
                dropped_total += dropped
                src = frames[num]
                stem = f"{seq}__{src.stem}"
                shutil.copy2(src, out_dir / "images" / sp / f"{stem}{src.suffix.lower()}")
                (out_dir / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                stats[sp]["images"] += 1
                stats[sp]["empty_images"] += int(not lines)
                for ln in lines:
                    stats[sp]["instances"][class_names[int(ln.split()[0])]] += 1
    for sp, st in stats.items():
        if st["images"] == 0:
            raise ValueError(f"Split '{sp}' has no images. Download more sequences / use a smaller stride.")

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": {i: n for i, n in enumerate(class_names)}}, sort_keys=False))
    report = {
        "seed": seed, "frame_stride": frame_stride, "merges": merges,
        "classes_found_in_annotations": dict(found), "training_classes": class_names,
        "split_sequences": splits, "dropped_degenerate_boxes": dropped_total,
        "splits": {sp: {"images": st["images"], "empty_images": st["empty_images"],
                        "instances": dict(st["instances"])} for sp, st in stats.items()},
        "notes": ["UA-DETRAC ignored regions are NOT masked (small label noise possible).",
                  "Split is by sequence to avoid frame leakage."],
    }
    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Convert UA-DETRAC subset to a YOLO dataset.")
    p.add_argument("--raw", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=YOLO_DIR)
    p.add_argument("--merge", action="append", help="merge classes, e.g. --merge van=car")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--frame-stride", type=int, default=1, help="extra thinning of local frames")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    try:
        rep = prepare(a.raw, a.out, parse_merge_args(a.merge), a.train_ratio, a.val_ratio,
                      a.seed, a.frame_stride, a.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"\nTraining classes: {rep['training_classes']}")
    for sp, st in rep["splits"].items():
        print(f"{sp:5s}: {st['images']:6d} images, instances {st['instances']}")
    print(f"\nWrote {a.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())