"""Streamlit-side helpers: page registry, navigation and per-browser-session settings."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st

from dashboard.session import SimulationSession, get_session as _get_session
from dashboard.sim_settings import default_settings

# (title, icon, module)  - every module exposes render()
PAGES: List[Tuple[str, str, str]] = [
    ("Dashboard", "📊", "dashboard.pages.dashboard_page"),
    ("Live Traffic Simulation", "🚦", "dashboard.pages.live_simulation_page"),
    ("AI Vehicle Detection", "🎥", "dashboard.pages.vehicle_detection_page"),
    ("DQN Controller", "🧠", "dashboard.pages.dqn_controller_page"),
    ("Emergency Priority", "🚑", "dashboard.pages.emergency_page"),
    ("Performance Comparison", "📈", "dashboard.pages.comparison_page"),
    ("Settings & Experiments", "⚙️", "dashboard.pages.settings_page"),
]


def nav_label(title: str) -> str:
    for t, icon, _ in PAGES:
        if t == title:
            return f"{icon}  {t}"
    raise KeyError(title)


def goto(title: str) -> None:
    """Use as a button callback: st.button(..., on_click=goto, args=("Live Traffic Simulation",))"""
    st.session_state["nav"] = nav_label(title)


def get_settings() -> Dict[str, Any]:
    if "settings" not in st.session_state:
        st.session_state["settings"] = default_settings()
    return st.session_state["settings"]


def save_settings(settings: Dict[str, Any]) -> None:
    st.session_state["settings"] = settings


def get_session() -> SimulationSession:
    return _get_session()