"""Download a bounded subset of the Hugging Face UA-DETRAC mirror.

Full UA-DETRAC is >100k frames. Consecutive frames are near-duplicates, so we download every
Nth frame of a limited number of sequences, plus ALL XML annotation files of those sequences.

Usage (from project root):
    python -m dataset.download_dataset --list
    python -m dataset.download_dataset --max-sequences 30 --frame-stride 10 --dry-run
    python -m dataset.download_dataset --max-sequences 30 --frame-stride 10
"""
from __future__ import annotations

import argparse
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.cv_config import HF_REPO_ID, RAW_DIR, SEQUENCE_REGEX
from utils.logging_utils import get_logger

log = get_logger(__name__)

IMG_EXT = (".jpg", ".jpeg", ".png")
SEQ_RE = re.compile(SEQUENCE_REGEX)
FRAME_RE = re.compile(r"(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def list_repo_files(repo_id: str = HF_REPO_ID) -> List[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed: pip install huggingface_hub") from exc
    try:
        return sorted(HfApi().list_repo_files(repo_id, repo_type="dataset"))
    except Exception as exc:
        raise RuntimeError(f"Could not list '{repo_id}' on Hugging Face (internet / repo name?): {exc}") from exc


def print_summary(files: List[str]) -> None:
    ext = Counter(Path(f).suffix.lower() or "<none>" for f in files)
    print(f"Total files: {len(files)}")
    print("By extension:", dict(ext.most_common()))
    tops = Counter("/".join(f.split("/")[:3]) for f in files)
    print("\nTop folders (first 3 path levels):")
    for k, v in tops.most_common(25):
        print(f"  {v:7d}  {k}")
    xml = [f for f in files if f.lower().endswith(".xml")]
    print(f"\nXML files: {len(xml)}  e.g. {xml[:3]}")
    txt = [f for f in files if f.lower().endswith(".txt")]
    if txt:
        print(f"TXT files (maybe YOLO labels?): {len(txt)}  e.g. {txt[:3]}")


def select_files(files: List[str], max_sequences: int, frame_stride: int,
                 seed: int) -> Tuple[List[str], List[str], int]:
    """Return (chosen_sequences, files_to_download, n_usable_sequences)."""
    images: Dict[str, List[str]] = defaultdict(list)
    xmls: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        m = SEQ_RE.search(f)
        if not m:
            continue
        low = f.lower()
        if low.endswith(IMG_EXT):
            images[m.group(0)].append(f)
        elif low.endswith(".xml"):
            xmls[m.group(0)].append(f)
    usable = sorted(set(images) & set(xmls))
    if not usable:
        raise RuntimeError(
            "No sequence has BOTH images and an XML annotation in the repository.\n"
            "Run with --list and send me the output; the repo may use another layout "
            "(for example ready-made YOLO .txt labels).")
    rng = random.Random(seed)
    shuffled = usable[:]
    rng.shuffle(shuffled)
    chosen = sorted(shuffled[:max_sequences])
    selected: List[str] = []
    for seq in chosen:
        for f in sorted(images[seq]):
            m = FRAME_RE.search(f)
            if m and (int(m.group(1)) - 1) % frame_stride == 0:
                selected.append(f)
        selected.extend(xmls[seq])
    return chosen, selected, len(usable)


def _fetch(filename: str, repo_id: str, local_dir: Path, retries: int = 3) -> Optional[str]:
    from huggingface_hub import hf_hub_download
    err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset",
                            local_dir=str(local_dir))
            return None
        except Exception as exc:  # network hiccup etc.
            err = exc
            time.sleep(1.5 * (attempt + 1))
    return f"{filename}: {err}"


def download(files: List[str], repo_id: str, local_dir: Path, workers: int = 8) -> int:
    from tqdm import tqdm
    local_dir.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    todo = [f for f in files if not (local_dir / f).exists()]
    log.info("%d files to fetch (%d already present).", len(todo), len(files) - len(todo))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_fetch, f, repo_id, local_dir) for f in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
            res = fut.result()
            if res:
                failures.append(res)
    if failures:
        log.error("%d downloads failed, e.g. %s", len(failures), failures[:3])
    return len(failures)


def main() -> int:
    p = argparse.ArgumentParser(description="Download a subset of the UA-DETRAC HF mirror.")
    p.add_argument("--repo", default=HF_REPO_ID)
    p.add_argument("--out", type=Path, default=RAW_DIR)
    p.add_argument("--list", action="store_true", help="print repository structure and exit")
    p.add_argument("--max-sequences", type=int, default=30)
    p.add_argument("--frame-stride", type=int, default=10, help="keep every Nth frame")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.max_sequences < 3 or args.frame_stride < 1:
        print("ERROR: --max-sequences must be >= 3 (train/val/test need separate sequences) "
              "and --frame-stride >= 1")
        return 1
    try:
        files = list_repo_files(args.repo)
        if args.list:
            print_summary(files)
            return 0
        chosen, selected, usable = select_files(files, args.max_sequences, args.frame_stride, args.seed)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    n_img = sum(1 for f in selected if f.lower().endswith(IMG_EXT))
    print(f"Usable annotated sequences in repo: {usable}; selected {len(chosen)}: {chosen[:5]} ...")
    print(f"Files to download: {len(selected)} ({n_img} images, {len(selected) - n_img} XML)")
    if args.dry_run:
        return 0
    failed = download(selected, args.repo, args.out, args.workers)
    print(f"Done. Data in {args.out}  (failures: {failed})")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())