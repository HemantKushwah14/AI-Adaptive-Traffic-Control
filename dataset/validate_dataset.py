"""Dataset inspection and validation.

    python -m dataset.validate_dataset --raw     # what did I download? which classes exist?
    python -m dataset.validate_dataset --yolo    # is the converted dataset valid / leak-free?
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import yaml
from PIL import Image

from config.cv_config import RAW_DIR, YOLO_DIR
from dataset.convert_annotations import parse_detrac_xml
from dataset.prepare_dataset import find_annotation_files, find_sequence_dirs, index_frames

IMG_EXT = (".jpg", ".jpeg", ".png")


def inspect_raw(raw_dir: Path) -> int:
    if not raw_dir.exists():
        print(f"[FAIL] {raw_dir} does not exist. Run: python -m dataset.download_dataset")
        return 1
    ext: Counter = Counter()
    for _r, _d, files in os.walk(raw_dir):
        ext.update(Path(f).suffix.lower() or "<none>" for f in files)
    print(f"File types under {raw_dir}: {dict(ext.most_common())}")
    seqs, xmls = find_sequence_dirs(raw_dir), find_annotation_files(raw_dir)
    usable = sorted(set(seqs) & set(xmls))
    print(f"Image sequences: {len(seqs)} | XML annotation files: {len(xmls)} | usable: {len(usable)}")
    if not usable:
        print("[FAIL] No sequence has both images and XML. Send me the output of "
              "`python -m dataset.download_dataset --list`.")
        return 1

    classes: Counter = Counter()
    frames_ann = frames_with_img = malformed = 0
    sizes: Counter = Counter()
    for s in usable:
        try:
            ann = parse_detrac_xml(xmls[s])
        except ValueError as exc:
            print(f"[WARN] {s}: {exc}")
            continue
        malformed += ann.malformed
        local = index_frames(seqs[s])
        for num, boxes in ann.frames.items():
            frames_ann += 1
            frames_with_img += int(num in local)
            classes.update(b.cls for b in boxes)
        if local:
            with Image.open(next(iter(local.values()))) as im:
                sizes[im.size] += 1
    print(f"Annotated frames: {frames_ann} | of which image present locally: {frames_with_img} "
          "(difference is expected when you downloaded every Nth frame)")
    print(f"Malformed XML entries skipped: {malformed}")
    print(f"Image sizes (per sequence): {dict(sizes)}")
    print("\nVehicle classes ACTUALLY present in the annotations (instances):")
    for k, v in classes.most_common():
        print(f"   {k:12s} {v}")
    missing = [c for c in ("car", "motorcycle", "bus", "truck") if c not in classes]
    if missing:
        print(f"\n[NOTE] Requested classes not present as labels: {missing}.\n"
              "       Fine-tuning cannot teach them. Use the pretrained COCO detector for these "
              "or add a supplementary dataset (see Part 3 notes).")
    print("\n[ OK ] Raw dataset inspection finished.")
    return 0


def validate_yolo(yolo_dir: Path, sample: int = 200) -> int:
    ypath = yolo_dir / "data.yaml"
    if not ypath.exists():
        print(f"[FAIL] {ypath} missing. Run: python -m dataset.prepare_dataset --overwrite")
        return 1
    cfg = yaml.safe_load(ypath.read_text())
    names: Dict[int, str] = cfg["names"]
    nc = len(names)
    problems: List[str] = []
    seqs: Dict[str, Set[str]] = {}
    inst: Dict[str, Counter] = {}
    all_imgs: List[Path] = []
    for sp in ("train", "val", "test"):
        idir, ldir = yolo_dir / "images" / sp, yolo_dir / "labels" / sp
        if not idir.exists() or not ldir.exists():
            problems.append(f"{sp}: missing images/labels folder")
            continue
        imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT)
        stems = {p.stem for p in imgs}
        seqs[sp] = {p.stem.split("__")[0] for p in imgs}
        inst[sp] = Counter()
        all_imgs += imgs
        for orphan in {p.stem for p in ldir.glob("*.txt")} - stems:
            problems.append(f"{sp}: label without image: {orphan}")
        for img in imgs:
            lbl = ldir / f"{img.stem}.txt"
            if not lbl.exists():
                problems.append(f"{sp}: missing label for {img.name}")
                continue
            for n, line in enumerate(lbl.read_text().splitlines(), 1):
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("expected 5 fields")
                    c = int(parts[0])
                    v = [float(x) for x in parts[1:]]
                    if not 0 <= c < nc:
                        raise ValueError(f"class id {c} outside 0..{nc - 1}")
                    if any(x < 0 or x > 1 for x in v) or v[2] <= 0 or v[3] <= 0:
                        raise ValueError("coordinates outside (0,1] range")
                    inst[sp][names[c]] += 1
                except ValueError as exc:
                    problems.append(f"{lbl.name}:{n}: {exc}")
        print(f"{sp:5s}: {len(imgs):6d} images | {len(seqs[sp]):3d} sequences | instances {dict(inst[sp])}")
        if len(imgs) == 0:
            problems.append(f"{sp}: no images")
        elif inst[sp].total() == 0:
            problems.append(f"{sp}: contains no labelled objects")

    sps = list(seqs)
    for i in range(len(sps)):
        for j in range(i + 1, len(sps)):
            overlap = seqs[sps[i]] & seqs[sps[j]]
            if overlap:
                problems.append(f"LEAKAGE: sequences in both {sps[i]} and {sps[j]}: {sorted(overlap)[:5]}")
    random.Random(0).shuffle(all_imgs)
    for img in all_imgs[:sample]:
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception as exc:
            problems.append(f"unreadable image {img.name}: {exc}")
    for sp in inst:
        for cid, cname in names.items():
            if sp == "train" and inst[sp][cname] < 50:
                print(f"[WARN] class '{cname}' has only {inst[sp][cname]} training instances.")

    if problems:
        print(f"\n[FAIL] {len(problems)} problem(s). First 20:")
        for p in problems[:20]:
            print("   -", p)
        return 1
    print("\n[ OK ] YOLO dataset is consistent, readable and leakage-free (by sequence).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--raw", action="store_true")
    g.add_argument("--yolo", action="store_true")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--yolo-dir", type=Path, default=YOLO_DIR)
    a = p.parse_args()
    return inspect_raw(a.raw_dir) if a.raw else validate_yolo(a.yolo_dir)


if __name__ == "__main__":
    raise SystemExit(main())