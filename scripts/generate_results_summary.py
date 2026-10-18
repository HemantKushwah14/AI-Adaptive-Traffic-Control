"""Collect YOUR measured results into one Markdown file (nothing is invented: every number is read
from files your own runs produced; missing items are listed with the command that creates them).

    python scripts\\generate_results_summary.py
    python scripts\\generate_results_summary.py --out docs\\RESULTS_SUMMARY.md

Colab metrics: copy vehicle_detector_test_metrics.json / ambulance_detector_test_metrics.json from
Google Drive into outputs\\vision\\ to include them.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.cv_config import VISION_OUT_DIR, YOLO_DIR  # noqa: E402
from config.rl_config import TRAIN_LOG, TRAIN_SUMMARY  # noqa: E402
from config.settings import OUTPUTS_DIR  # noqa: E402
from evaluation.emergency_report import load_latest_emergency  # noqa: E402
from evaluation.reports import df_to_markdown, load_latest  # noqa: E402


def missing(what: str, command: str) -> List[str]:
    return [f"_Not available: {what}. Create it with_ `{command}`", ""]


def section_dataset() -> List[str]:
    out = ["## Vehicle dataset (converted UA-DETRAC subset)", ""]
    p = YOLO_DIR / "prepare_report.json"
    if not p.exists():
        return out + missing("no converted dataset", "python -m dataset.prepare_dataset --overwrite")
    r = json.loads(p.read_text(encoding="utf-8"))
    out += [f"- Classes found in the annotations: {r['classes_found_in_annotations']}",
            f"- Training classes: {r['training_classes']} (merges: {r['merges']})",
            f"- Split by sequence (seed {r['seed']}): "
            + ", ".join(f"{k}: {len(v)} seq." for k, v in r["split_sequences"].items())]
    out += [f"- {k}: {v['images']} images, instances {v['instances']}" for k, v in r["splits"].items()]
    return out + [""]


def section_detector(title: str, file: str) -> List[str]:
    out = [f"## {title}", ""]
    p = VISION_OUT_DIR / file
    if not p.exists():
        return out + missing(f"{file} (output of the Colab notebook, held-out TEST split)",
                             f"copy {file} from Google Drive into outputs\\vision\\")
    m = json.loads(p.read_text(encoding="utf-8"))
    out += [f"- Split: {m.get('split')} | epochs {m.get('epochs')} | image size {m.get('imgsz')}",
            f"- mAP@50: {m['map50']:.3f} | mAP@50-95: {m['map50_95']:.3f} | precision: {m['precision']:.3f} | recall: {m['recall']:.3f}"]
    if m.get("per_class_map50_95"):
        out.append("- Per-class mAP@50-95: " + ", ".join(f"{k} {v:.3f}" for k, v in m["per_class_map50_95"].items()))
    if m.get("caveat"):
        out.append(f"- Caveat: {m['caveat']}")
    return out + [""]


def section_dqn() -> List[str]:
    out = ["## DQN training", ""]
    if not TRAIN_SUMMARY.exists():
        return out + missing("no training summary", "python -m reinforcement_learning.train --episodes 80")
    s = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
    out += [f"- Episodes completed: {s['episodes_completed']} | stored transitions: {s['env_steps']} | learning steps: {s['learn_steps']}",
            f"- Best validation average waiting time (validation seeds): {s['best_val_wait_s']}",
            f"- Fixed-time on the same validation seeds: {s['val_fixed_wait_s']}",
            f"- Wall-clock training time: {s['wall_clock_s'] / 60:.1f} min", f"- Note: {s['note']}"]
    if TRAIN_LOG.exists():
        df = pd.read_csv(TRAIN_LOG)
        if len(df):
            last = df.iloc[-1]
            out.append(f"- Last logged episode {int(last['episode'])}: reward {last['total_reward']:.1f}, "
                       f"epsilon {last['epsilon']:.2f}, mean loss {last['mean_loss']:.4f}")
    return out + [""]


def section_comparison() -> List[str]:
    out = ["## Fixed-time vs DQN (held-out seeds, identical demand)", ""]
    try:
        data = load_latest()
    except ValueError as exc:
        return out + [f"_Saved comparison unreadable: {exc}_", ""]
    if data is None:
        return out + missing("no comparison report", "python scripts\\run_comparison.py --seeds 5")
    rep, cfg = data["report"], data["report"]["config"]
    out += [f"- Generated: {rep.get('generated_at')} | seeds: {rep['seeds']} | {cfg['duration']} s per run",
            f"- Demand (veh/h): {cfg['demand_vph']} | green limits {cfg['min_green']}-{cfg['max_green']} s",
            f"- Fixed-time plan (s): {rep['fixed_greens']} | DQN checkpoint: {rep['checkpoint'].get('checkpoint')} "
            f"({rep['checkpoint'].get('episodes_trained')} training episodes)", ""]
    out += [f"> WARNING: {w}" for w in rep.get("warnings", [])]
    out += ["Improvement is positive when the DQN is better.", "", df_to_markdown(data["improvement"].round(2)), "",
            "Per-seed improvement %:", "", df_to_markdown(data["per_seed"].round(1)), "",
            "Context (mean over seeds):", "", df_to_markdown(data["context"].round(1)), ""]
    return out


def section_emergency() -> List[str]:
    out = ["## Emergency priority (simulated ambulances, simulation ground-truth detection)", ""]
    try:
        data = load_latest_emergency()
    except ValueError as exc:
        return out + [f"_Saved scenario unreadable: {exc}_", ""]
    if data is None or "ambulance_comparison" not in data:
        return out + missing("no emergency scenario", "python scripts\\run_emergency_demo.py")
    rep = data["report"]
    out += [f"- Base controller: {rep.get('base')} | scheduled ambulances: {rep.get('events_scheduled')}",
            f"- Detection source: {rep.get('detection_source')}", "",
            df_to_markdown(data["ambulance_comparison"].round(1), index=False), ""]
    if "kpi_comparison" in data:
        out += ["Cost/benefit for overall traffic (priority minus no priority):", "",
                df_to_markdown(data["kpi_comparison"].round(2), index=False), ""]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUTPUTS_DIR / "RESULTS_SUMMARY.md")
    a = p.parse_args()
    lines = ["# Results summary", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} from files produced by your own runs. "
             "Nothing in this file is hand-written or pre-filled.", ""]
    lines += section_dataset() + section_detector("Vehicle detector (Colab, test split)", "vehicle_detector_test_metrics.json")
    lines += section_detector("Ambulance detector (Colab, test split)", "ambulance_detector_test_metrics.json")
    lines += section_dqn() + section_comparison() + section_emergency()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())