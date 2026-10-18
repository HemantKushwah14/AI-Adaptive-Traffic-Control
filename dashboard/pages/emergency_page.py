"""Page 5 - Emergency Priority. Live status from the running session, a reproducible
with/without-priority scenario (background job), and the vision ambulance detector (only if
you trained one). Simulated ambulances and ground-truth detection are labelled as such."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from computer_vision.ambulance_detector import (HELP, LIMITATIONS, AmbulanceDetector,
                                                AmbulanceDetectorUnavailableError, locate_sightings)
from computer_vision.detector import draw_detections
from computer_vision.lane_analyzer import LaneAnalyzer, load_or_default
from computer_vision.video_processor import decode_image_bytes
from config.constants import APPROACH_NAMES, LANE_INDEX, PHASE_NAMES
from config.emergency_config import (DEFAULT_CLEARANCE_BUFFER_S, DEFAULT_DETECTION_RANGE_M,
                                     DEFAULT_MAX_HOLD_S, DEFAULT_SCENARIO)
from dashboard.components import ui
from dashboard.jobs import JobState, emergency_target, format_events, get_emergency_job, parse_events
from dashboard.session import SessionState
from dashboard.sim_settings import build_sim_config
from dashboard.state import get_session, get_settings
from dashboard.styles.theme import COLORS
from dashboard.visualizations import charts
from evaluation.emergency_report import load_latest_emergency
from reinforcement_learning.dqn_agent import CheckpointError
from simulation.adaptive_controller import default_checkpoint_path
from simulation.emergency_controller import required_phase

MODES = ["NORMAL", "PREEMPTING", "EMERGENCY_GREEN", "RECOVERY"]
MODE_TONE = {"NORMAL": "green", "PREEMPTING": "amber", "EMERGENCY_GREEN": "red", "RECOVERY": "blue"}


@st.cache_resource(show_spinner=False)
def _amb_detector() -> AmbulanceDetector:
    d = AmbulanceDetector()
    d.load()
    return d


# ------------------------------------------------------------------ live status
def _live_status(sess) -> None:
    snap = sess.snapshot()
    if snap is None:
        ui.alert("No simulation is running. Start one on the Live page and dispatch a simulated ambulance, or run the "
                 "reproducible scenario below.", "blue", "Live emergency status")
        return
    em = snap["emergency"]
    ui.md('<div class="muted">Mode of the emergency state machine</div>'
          + ui.phase_chips(MODES, MODES.index(em["mode"])))
    reqs = em["active_requests"]
    first = reqs[0] if reqs else None
    prog = em["clearance_progress"]
    recov = em["recovery_remaining_s"]
    ui.kpi_grid([
        dict(label="Ambulance detection", value="DETECTED" if reqs else "None",
             sub=(f'{len(reqs)} active · source: simulation ground truth (SIMULATED)' if reqs else "no emergency vehicle in range"),
             accent="red" if reqs else "green", icon="🚑", pulse=bool(reqs)),
        dict(label="Emergency approach", value=ui.esc(APPROACH_NAMES[em["target_approach"]]) if em["target_approach"] else "—",
             sub=ui.esc(f'needs {em["target_phase"]}') if em["target_phase"] else "", accent="amber", icon="🧭"),
        dict(label="Priority status", value=ui.esc(em["mode"].replace("_", " ").title()) if em["enabled"] else "Disabled",
             sub=ui.pill(f'{em["activations"]} activation(s) · {em["timeouts"]} timeout(s)', MODE_TONE[em["mode"]]),
             accent="red" if em["mode"] != "NORMAL" else "green", icon="🛡"),
        dict(label="Signal transition", value=ui.esc(em["signal_transition"]),
             sub=f'{em["transition_remaining_s"]:.0f} s remaining' if em["transition_remaining_s"] else "no clearance in progress",
             accent="purple", icon="🔀"),
        dict(label="Clearance progress", value="—" if prog is None else f"{prog * 100:.0f}%",
             sub="distance-based estimate", accent="cyan", icon="📏"),
        dict(label="Return to normal control", value="Pending" if em["mode"] != "NORMAL" else "Normal",
             sub=(f'restoring {ui.esc(em["base_controller"])} in {recov:.0f} s' if recov > 0 else f'base: {ui.esc(em["base_controller"])}'),
             accent="blue", icon="↩")], cols=3)
    if prog is not None:
        st.progress(float(min(max(prog, 0.0), 1.0)), text="Emergency clearance progress")
    if reqs:
        rows = [(ui.esc(r["vehicle_id"]), APPROACH_NAMES[r["approach"]], ui.esc(r["required_phase"]), ui.esc(r["status"]),
                 f'{r["detected_time_s"]:.0f} s', f'{r["progress"] * 100:.0f}%') for r in reqs]
        ui.md(ui.card("Active emergency requests", ui.html_table(["Vehicle", "Approach", "Needs", "Status", "Detected at", "Progress"], rows)))
    log = "".join(f'<div class="logline">t={m["time"]:.0f}s · {ui.esc(m["message"])}</div>' for m in em["log"])
    if log:
        ui.md(ui.card("Event log", log))
    ev = sess.emergency_events_df()
    if len(ev):
        st.dataframe(ev, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ scenario job
def _job_progress(job) -> None:
    running = job.snapshot()["state"] == JobState.RUNNING.value

    @st.fragment(run_every=1.0 if running else None)
    def view() -> None:
        s = job.snapshot()
        if running and s["state"] != JobState.RUNNING.value:
            st.rerun()
        if s["state"] == JobState.RUNNING.value:
            st.progress(min(s["done"] / max(s["total"], 1), 1.0), text=f'{s["message"]} · {s["elapsed_s"]:.0f} s elapsed')
            if st.button("Cancel (after current run)", key="em_cancel"):
                job.cancel()
        elif s["state"] == JobState.ERROR.value:
            ui.alert(s["error"], "red", "Scenario failed", preformatted=True)
        elif s["state"] == JobState.CANCELLED.value:
            ui.alert("The scenario was cancelled.", "amber")

    view()


def _scenario_runner(settings: dict) -> None:
    job = get_emergency_job()
    with st.container(border=True):
        c = st.columns([2.4, 1, 1, 1.2])
        ev_text = c[0].text_input("Simulated ambulances (time:approach:movement, comma-separated)",
                                  value=format_events(DEFAULT_SCENARIO), key="em_events")
        duration = c[1].number_input("Duration (s)", 120, 3600, 600, 60, key="em_dur")
        seed = c[2].number_input("Seed", 0, 1_000_000, int(settings["seed"]), key="em_seed")
        base_kind = c[3].selectbox("Base controller", ["fixed", "dqn"], index=1 if settings["controller"] == "dqn" else 0,
                                   format_func=lambda k: "Fixed-time" if k == "fixed" else "DQN", key="em_base")
        d = st.columns(3)
        rng = d[0].slider("Detection range (m)", 50, 250, int(DEFAULT_DETECTION_RANGE_M), 10, key="em_rng")
        buf = d[1].slider("Clearance buffer (s)", 0, 10, int(DEFAULT_CLEARANCE_BUFFER_S), key="em_buf")
        hold = d[2].slider("Max emergency hold (s)", 10, 90, int(DEFAULT_MAX_HOLD_S), 5, key="em_hold")
        running = job.snapshot()["state"] == JobState.RUNNING.value
        if st.button("▶  Run reproducible scenario (with vs without priority)", type="primary", disabled=running, key="em_run"):
            try:
                cfg = dataclasses.replace(build_sim_config(settings), duration=int(duration), seed=int(seed),
                                          emergency_events=parse_events(ev_text), emergency_probability=0.0,
                                          step_length=1.0)
                cfg.validate()
                ckpt = None
                if base_kind == "dqn":
                    ck = settings.get("checkpoint")
                    ckpt = Path(ck) if ck else default_checkpoint_path()
                job.start(emergency_target, cfg, base_kind, ckpt, tuple(settings["fixed_greens"]), float(rng), float(buf), float(hold))
                st.rerun()
            except (ValueError, CheckpointError) as exc:
                ui.alert(str(exc), "red", "Cannot start the scenario", preformatted=True)
        st.caption("Ambulances are explicitly SIMULATED SUMO vehicles; detection is simulation ground truth read through TraCI. "
                   "Both runs use identical demand and seed (route files are hash-verified).")
    _job_progress(job)


def _scenario_results() -> None:
    try:
        data = load_latest_emergency()
    except ValueError as exc:
        ui.alert(str(exc), "red", "Saved scenario unreadable")
        return
    if data is None or "ambulance_comparison" not in data:
        st.caption("No saved scenario yet. Run it above or: python scripts\\run_emergency_demo.py")
        return
    rep = data["report"]
    ui.alert(f'Base controller: {rep.get("base")} · duration {rep["config"]["duration"]} s · seed {rep["config"]["seed"]} · '
             f'scheduled: {rep.get("events_scheduled")} · detection source: {rep.get("detection_source")}', "blue", "Latest saved scenario")
    amb = data["ambulance_comparison"]
    if len(amb):
        st.dataframe(amb.round(1), use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=amb["vehicle_id"], y=amb["clear_s_no_priority"], name="No priority", marker_color=COLORS["amber"]))
        fig.add_trace(go.Bar(x=amb["vehicle_id"], y=amb["clear_s_priority"], name="With priority", marker_color=COLORS["cyan"]))
        fig.update_layout(barmode="group")
        fig = charts.style(fig, 280, "Detection-to-clear time per ambulance (s)")
        fig.update_xaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG, key="em_bar")
        if (amb["time_saved_s"].fillna(0) <= 0).any():
            st.caption("A time saved of 0 or below is a real outcome: the ambulance may already have had green, or the base controller was already quick.")
    else:
        ui.alert("No ambulance was detected in the scenario (check event times and detection range).", "amber")
    if "kpi_comparison" in data:
        st.markdown("**Cost to overall traffic (priority minus no priority; positive difference on delay/queue = worse)**")
        st.dataframe(data["kpi_comparison"].round(2), use_container_width=True, hide_index=True)
    tl = data.get("timeline_priority")
    if tl is not None and len(tl):
        lvl = tl["mode"].map({m: i for i, m in enumerate(MODES)})
        fig = go.Figure(go.Scatter(x=tl["time"], y=lvl, mode="lines", line=dict(shape="hv", color=COLORS["red"], width=2),
                                   fill="tozeroy", fillcolor="rgba(255,77,94,0.10)"))
        fig.update_yaxes(tickvals=list(range(len(MODES))), ticktext=MODES, range=[-0.3, len(MODES) - 0.7])
        st.plotly_chart(charts.style(fig, 230, "Emergency state machine (with priority)", legend=False),
                        use_container_width=True, config=charts.PLOT_CONFIG, key="em_mode")
        st.plotly_chart(charts.phase_timeline(tl.rename(columns={"active_phase": "phase"})),
                        use_container_width=True, config=charts.PLOT_CONFIG, key="em_phase")
    msgs = rep.get("messages", [])
    if msgs:
        ui.md(ui.card("Event log (with priority)", "".join(
            f'<div class="logline">t={m["time"]:.0f}s · {ui.esc(m["message"])}</div>' for m in msgs)))
    d = st.columns(2)
    if "ambulance_comparison" in data:
        d[0].download_button("⬇  Ambulance comparison (CSV)", data["ambulance_comparison"].to_csv(index=False).encode("utf-8"),
                             file_name="ambulance_comparison.csv", mime="text/csv", use_container_width=True)
    if "kpi_comparison" in data:
        d[1].download_button("⬇  Traffic KPI comparison (CSV)", data["kpi_comparison"].to_csv(index=False).encode("utf-8"),
                             file_name="emergency_kpi_comparison.csv", mime="text/csv", use_container_width=True)


# ------------------------------------------------------------------ vision detector
def _vision_section() -> None:
    with st.expander("📷  Ambulance detection from an image (separately trained model)"):
        ui.alert("  ·  ".join(LIMITATIONS), "amber", "Limitations of visual ambulance detection")
        if not AmbulanceDetector().available():
            ui.alert(HELP.format(path="models\\ambulance_detector.pt"), "blue",
                     "Ambulance model not trained yet (no substitute is used)", preformatted=True)
            return
        up = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"], key="amb_up")
        conf = st.slider("Confidence threshold", 0.30, 0.90, 0.50, 0.05, key="amb_conf")
        if st.button("Detect ambulances", disabled=up is None, key="amb_run"):
            try:
                det = _amb_detector()
                det.conf = conf
                analyzer = LaneAnalyzer(load_or_default())
                img = decode_image_bytes(up.getvalue())
                dets = det.predict(img)
                sights = locate_sightings(dets, img.shape, analyzer)
                st.image(draw_detections(analyzer.draw_overlay(img), dets), channels="BGR", use_column_width=True)
                if not sights:
                    st.info("No ambulance detected above the threshold.")
                rows = []
                for s in sights:
                    ph = "—"
                    if s.approach:
                        ph = PHASE_NAMES[required_phase(s.approach, LANE_INDEX["straight"] if s.lane_index is None else s.lane_index)]
                    rows.append({"confidence": round(s.confidence, 3), "approach": s.approach or "outside regions",
                                 "lane": s.lane_index, "would request phase": ph})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if sights and not sights[0].calibrated:
                    ui.alert("Regions are uncalibrated: the approach above is not meaningful.", "amber")
                st.caption("A single frame is not confirmed (the controller requires several confirming frames). This result is NOT "
                           "connected to the running SUMO scene, which has no camera.")
            except (AmbulanceDetectorUnavailableError, ValueError) as exc:
                ui.alert(str(exc), "red", "Detection failed", preformatted=True)


def render() -> None:
    ui.page_header("Emergency Priority", "Ambulance pre-emption: detection, safe signal transition, clearance and restoration of normal control.", "🚑")
    sess, settings = get_session(), get_settings()
    polling = sess.state == SessionState.RUNNING

    @st.fragment(run_every=1.0 if polling else None)
    def live() -> None:
        if polling and sess.state != SessionState.RUNNING:
            st.rerun()
        _live_status(sess)

    live()
    st.markdown("### Reproducible scenario")
    _scenario_runner(settings)
    _scenario_results()
    st.markdown("### Vision-based ambulance detection")
    _vision_section()