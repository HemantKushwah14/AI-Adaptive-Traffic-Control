"""Save/load emergency-demo results and build the with/without-priority ambulance table."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.emergency_config import EMERGENCY_OUT_DIR


def ambulance_comparison(off: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:
    """Per ambulance: detection->clear time and accumulated waiting, without vs with priority."""
    a = off[["vehicle_id", "approach", "required_phase", "detection_to_clear_s", "accum_waiting_s"]].rename(
        columns={"detection_to_clear_s": "clear_s_no_priority", "accum_waiting_s": "wait_s_no_priority"})
    b = on[["vehicle_id", "detection_to_clear_s", "accum_waiting_s", "wait_for_green_s", "status"]].rename(
        columns={"detection_to_clear_s": "clear_s_priority", "accum_waiting_s": "wait_s_priority",
                 "wait_for_green_s": "green_after_detection_s", "status": "status_priority"})
    m = a.merge(b, on="vehicle_id", how="outer")
    m["time_saved_s"] = m["clear_s_no_priority"] - m["clear_s_priority"]
    return m


def save_emergency_report(report: dict, frames: Dict[str, pd.DataFrame],
                          out_root: Optional[Path] = None) -> Path:
    root = Path(out_root) if out_root else EMERGENCY_OUT_DIR
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(run_dir / f"{name}.csv", index=False)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    latest = root / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(run_dir, latest)
    return run_dir


def load_latest_emergency(root: Optional[Path] = None) -> Optional[Dict]:
    d = (Path(root) if root else EMERGENCY_OUT_DIR) / "latest"
    if not (d / "report.json").exists():
        return None
    try:
        out = {"dir": d, "report": json.loads((d / "report.json").read_text(encoding="utf-8"))}
        for p in d.glob("*.csv"):
            out[p.stem] = pd.read_csv(p)
        return out
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Saved emergency report in {d} is unreadable: {exc}") from exc