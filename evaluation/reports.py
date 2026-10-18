"""Save / load comparison reports (CSV, JSON, Markdown) and helpers for CSV download."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.settings import OUTPUTS_DIR
from evaluation.compare import ComparisonResult

COMPARISON_DIR = OUTPUTS_DIR / "comparison"


def _fmt(v, digits: int = 2) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{digits}f}"
    return str(v)


def df_to_markdown(df: pd.DataFrame, digits: int = 2, index: bool = True) -> str:
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + [_fmt(v, digits) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    """For st.download_button."""
    return df.to_csv(index=index).encode("utf-8")


def build_markdown(result: ComparisonResult, generated_at: Optional[str] = None) -> str:
    cfg = result.config
    imp = result.improvement_table()
    ck = result.checkpoint
    lines = [
        "# Fixed-time vs DQN traffic signal control - comparison report", "",
        f"Generated: {generated_at or datetime.now().isoformat(timespec='seconds')}", "",
        "## Setup", "",
        f"- Simulated duration per run: {cfg['duration']} s",
        f"- Demand (veh/h per approach): {cfg['demand_vph']}",
        f"- Turn probabilities: {cfg['turn_probs']}",
        f"- Vehicle mix: {cfg['vehicle_mix']}",
        f"- Min/max green: {cfg['min_green']}/{cfg['max_green']} s; yellow {cfg['yellow_time']} s; "
        f"all-red {cfg['all_red_time']} s",
        f"- Seeds (identical for both controllers, route files hash-verified): {result.seeds}",
        f"- Fixed-time green plan (phases NS_THROUGH, NS_LEFT, EW_THROUGH, EW_LEFT): {result.fixed_greens} s",
        f"- DQN checkpoint: `{ck.get('checkpoint')}` | episodes trained: {ck.get('episodes_trained')} | "
        f"trained at: {ck.get('trained_at')}",
        "", "## Results (mean over seeds, measured from SUMO via TraCI)", "",
        "Improvement % is positive when the DQN is better (higher throughput, lower everything else).", "",
        df_to_markdown(imp), "",
        "## Context (mean over seeds)", "", df_to_markdown(result.context_table()), "",
        "## Per-seed improvement % (positive = DQN better)", "",
        df_to_markdown(result.per_seed_improvement()), "",
    ]
    if result.warnings:
        lines += ["## Compatibility warnings", ""] + [f"- {w}" for w in result.warnings] + [""]
    lines += [
        "## Limitations", "",
        "- Results apply only to the demand, duration and seeds listed above.",
        "- The fixed-time plan is a hand-chosen baseline, not an optimised one; a stronger baseline "
        "may reduce the measured improvement.",
        "- Average travel time covers only vehicles that arrived; check the context table for "
        "vehicles still in the network.",
        "- No statistical significance test is performed; read the 'DQN better in seeds' column "
        "for consistency.",
        "- The DQN was trained on simulated SUMO traffic only.", ""]
    return "\n".join(lines)


def save_report(result: ComparisonResult, out_root: Optional[Path] = None) -> Dict[str, Path]:
    root = Path(out_root) if out_root else COMPARISON_DIR
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().isoformat(timespec="seconds")

    paths = {
        "summaries": run_dir / "summaries.csv", "timeseries": run_dir / "timeseries.csv",
        "improvement": run_dir / "improvement.csv", "per_seed": run_dir / "per_seed_improvement.csv",
        "context": run_dir / "context.csv", "json": run_dir / "report.json",
        "markdown": run_dir / "report.md"}
    result.summaries.to_csv(paths["summaries"], index=False)
    result.timeseries.to_csv(paths["timeseries"], index=False)
    result.improvement_table().to_csv(paths["improvement"])
    result.per_seed_improvement().to_csv(paths["per_seed"])
    result.context_table().to_csv(paths["context"])
    paths["json"].write_text(json.dumps({"generated_at": generated, **result.to_dict()},
                                        indent=2, default=str), encoding="utf-8")
    paths["markdown"].write_text(build_markdown(result, generated), encoding="utf-8")

    latest = root / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(run_dir, latest)
    paths["dir"] = run_dir
    return paths


def load_latest(root: Optional[Path] = None) -> Optional[Dict]:
    """Load the most recent saved comparison (used by the dashboard). None if nothing saved yet."""
    d = (Path(root) if root else COMPARISON_DIR) / "latest"
    if not (d / "report.json").exists():
        return None
    try:
        return {
            "dir": d,
            "report": json.loads((d / "report.json").read_text(encoding="utf-8")),
            "summaries": pd.read_csv(d / "summaries.csv"),
            "timeseries": pd.read_csv(d / "timeseries.csv"),
            "improvement": pd.read_csv(d / "improvement.csv", index_col=0),
            "per_seed": pd.read_csv(d / "per_seed_improvement.csv", index_col=0),
            "context": pd.read_csv(d / "context.csv", index_col=0),
            "markdown": (d / "report.md").read_text(encoding="utf-8"),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Saved comparison in {d} is unreadable: {exc}") from exc