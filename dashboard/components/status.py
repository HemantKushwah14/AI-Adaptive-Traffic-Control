"""System readiness checks shown on the Dashboard (cached 10 s). Reports facts, never assumes."""
from __future__ import annotations

import json
from typing import Dict, List

import streamlit as st

from config.cv_config import (AMBULANCE_WEIGHTS, LANE_REGION_FILE, PRETRAINED_WEIGHTS, VEHICLE_WEIGHTS)
from config.rl_config import BEST_CKPT, LAST_CKPT
from config.settings import NET_FILE, SumoNotFoundError, ensure_traci_importable, find_sumo_binary
from evaluation.reports import COMPARISON_DIR


def _item(key: str, label: str, tone: str, detail: str) -> dict:
    return {"key": key, "label": label, "tone": tone, "detail": detail}


@st.cache_data(ttl=10, show_spinner=False)
def system_status() -> List[Dict]:
    out: List[Dict] = []
    try:
        find_sumo_binary("sumo")
        ensure_traci_importable()
        out.append(_item("sumo", "SUMO + TraCI", "ok", "found"))
    except SumoNotFoundError:
        out.append(_item("sumo", "SUMO + TraCI", "bad", "not found - see setup help"))
    out.append(_item("net", "SUMO network", "ok" if NET_FILE.exists() else "warn",
                     "built" if NET_FILE.exists() else "built automatically on first Start"))

    ck = BEST_CKPT if BEST_CKPT.exists() else (LAST_CKPT if LAST_CKPT.exists() else None)
    if ck is None:
        out.append(_item("dqn", "DQN checkpoint", "warn", "none yet - train (Part 4) or use fixed-time"))
    else:
        try:
            from reinforcement_learning.dqn_agent import read_checkpoint_meta
            m = read_checkpoint_meta(ck)
            out.append(_item("dqn", "DQN checkpoint", "ok",
                             f"{ck.name}: {m['episodes']} training episodes (quality: see Page 6)"))
        except Exception as exc:
            out.append(_item("dqn", "DQN checkpoint", "bad", f"unreadable: {exc}"))

    if VEHICLE_WEIGHTS.exists():
        out.append(_item("detector", "Vehicle detector", "ok", "fine-tuned weights present"))
    elif PRETRAINED_WEIGHTS.exists():
        out.append(_item("detector", "Vehicle detector", "warn", "pretrained COCO only (not fine-tuned)"))
    else:
        out.append(_item("detector", "Vehicle detector", "warn",
                         "no weights yet; first detection downloads yolov8n.pt"))
    out.append(_item("amb", "Ambulance detector", "ok" if AMBULANCE_WEIGHTS.exists() else "warn",
                     "weights present" if AMBULANCE_WEIGHTS.exists() else "not trained; SUMO demo uses simulated ambulances"))
    try:
        cal = bool(json.loads(LANE_REGION_FILE.read_text()).get("calibrated")) if LANE_REGION_FILE.exists() else False
    except (OSError, ValueError):
        cal = False
    out.append(_item("regions", "Camera lane regions", "ok" if cal else "warn",
                     "calibrated" if cal else "uncalibrated (per-approach vision counts not meaningful)"))
    has_cmp = (COMPARISON_DIR / "latest" / "report.json").exists()
    out.append(_item("compare", "Comparison report", "ok" if has_cmp else "warn",
                     "available" if has_cmp else "none yet - run scripts\\run_comparison.py"))
    return out