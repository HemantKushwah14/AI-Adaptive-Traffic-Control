"""Streamlit dashboard entry point for AdaptiveTrafficRL."""
from __future__ import annotations

import importlib

import streamlit as st

from config.settings import ensure_dirs
from dashboard.components import ui
from dashboard.state import PAGES, nav_label
from dashboard.styles.theme import inject_css

st.set_page_config(
    page_title="Adaptive Traffic RL",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_dirs()
inject_css()

labels = [nav_label(title) for title, _, _ in PAGES]
if "nav" not in st.session_state:
    st.session_state["nav"] = labels[0]

with st.sidebar:
    ui.md(
        '<div class="brand">'
        '<div class="brand-logo">🚦</div>'
        '<div><div class="brand-name">Adaptive Traffic RL</div>'
        '<div class="brand-sub">SUMO · DQN · Vision</div></div></div>'
    )
    st.radio("Navigation", labels, label_visibility="collapsed", key="nav")
    ui.md('<div class="muted" style="margin-top:18px">Live simulation runs in a background thread; '
          'reruns do not restart SUMO.</div>')

choice = st.session_state["nav"]
for title, icon, module_name in PAGES:
    if choice == nav_label(title):
        importlib.import_module(module_name).render()
        break
