"""Page 1 - Dashboard: overview, system readiness, live KPI cards and charts."""
from __future__ import annotations

import streamlit as st

from dashboard.components import ui
from dashboard.components.status import system_status
from dashboard.session import SessionState
from dashboard.sim_settings import CONTROLLERS
from dashboard.state import get_session, goto
from dashboard.visualizations import charts

STATE_TONE = {"IDLE": ("blue", "Idle"), "RUNNING": ("green", "Running"), "PAUSED": ("amber", "Paused"),
              "FINISHED": ("cyan", "Finished"), "STOPPED": ("amber", "Stopped"), "ERROR": ("red", "Error")}
DOT = {"ok": "ok", "warn": "warn", "bad": "bad"}


def _overview() -> None:
    left, right = st.columns([1.35, 1])
    with left:
        body = (
            '<div style="font-size:15px;line-height:1.55;color:#cfe0ff">An intelligent four-way intersection simulated in '
            '<b>Eclipse SUMO</b>. A <b>Deep Q-Network</b> chooses the next signal phase, the choice is applied to the real '
            'SUMO traffic light through <b>TraCI</b> (with yellow and all-red safety transitions), and an '
            '<b>emergency-priority controller</b> pre-empts the signal for ambulances.</div>'
            + ui.flow(["Camera / video", "YOLO detection", "Lane density", "SUMO state", "DQN", "Phase + duration",
                       "SUMO / TraCI", "Metrics", "Dashboard"])
            + '<div class="muted" style="margin-top:12px">In this build the DQN is trained and run on <b>SUMO-measured state</b>. '
              'The vision module analyses uploaded images/videos (Page 3). Every KPI on this page is measured from the running '
              'simulation; nothing is pre-filled.</div>')
        ui.md(ui.card("Project overview", body))
        st.button("▶  Open Live Simulation", type="primary", on_click=goto, args=("Live Traffic Simulation",))
    with right:
        rows = [(f'<span class="dot {DOT[s["tone"]]}"></span>{ui.esc(s["label"])}',
                 f'<span class="muted">{ui.esc(s["detail"])}</span>') for s in system_status()]
        ui.md(ui.card("System readiness", ui.html_table(["Component", "Status"], rows)))


def _kpis(snap, status: dict) -> None:
    dq = status.get("dqn", {})
    dqn_tone = {"ok": "green", "warn": "amber", "bad": "red"}[dq.get("tone", "warn")]
    if snap is None:
        state_card = dict(label="Simulation status", value="Idle", sub="Press Start on the Live page", accent="blue", icon="⏻")
        k = None
    else:
        tone, label = STATE_TONE[snap["state"]]
        ctrl = CONTROLLERS.get(snap["controller"], snap["controller"])
        state_card = dict(label="Simulation status", value=label,
                          sub=f't = {snap["time"]:.0f} / {snap["duration"]} s · {ui.esc(ctrl)}',
                          accent={"green": "green", "amber": "amber", "red": "red", "cyan": "cyan", "blue": "blue"}[tone], icon="⏻")
        k = snap["kpi"]
    na = "—"
    cards = [state_card]
    if k is None:
        cards += [dict(label="Total vehicles", value=na, sub="entered the network", accent="cyan", icon="🚗"),
                  dict(label="Avg waiting time", value=na, sub="per vehicle", accent="blue", icon="⏱"),
                  dict(label="Throughput", value=na, sub="trailing 60 s", accent="green", icon="🏁"),
                  dict(label="Avg queue length", value=na, sub="halting vehicles", accent="amber", icon="🚧"),
                  dict(label="Active phase", value=na, sub="", accent="purple", icon="🚥"),
                  dict(label="Emergency vehicle", value=na, sub="", accent="green", icon="🚑")]
    else:
        sg, em = snap["signal"], snap["emergency"]
        cd = snap["countdown"]
        cards += [
            dict(label="Total vehicles", value=ui.fmt(k["vehicles_entered"], 0),
                 sub=f'{k["vehicles_in_network"]} in network · {k["vehicles_arrived"]} arrived', accent="cyan", icon="🚗"),
            dict(label="Avg waiting time", value=ui.fmt(k["avg_waiting_time_s"], 1, " s"),
                 sub="accumulated, per vehicle", accent="blue", icon="⏱"),
            dict(label="Throughput", value=ui.fmt(k["throughput_now_vph"], 0, " veh/h"),
                 sub=f'trailing 60 s · run avg {ui.fmt(k["throughput_avg_vph"], 0)} veh/h', accent="green", icon="🏁"),
            dict(label="Avg queue length", value=ui.fmt(k["avg_queue_length"], 1, " veh"),
                 sub=f'now {k["queue_now"]:.0f} halting', accent="amber", icon="🚧"),
            dict(label="Active phase", value=ui.esc(sg["phase_names"][sg["active_phase"]]),
                 sub=f'{ui.pill(sg["mode"].replace("_", "-"), {"GREEN": "green", "YELLOW": "amber", "ALL_RED": "red"}[sg["mode"]])}'
                     f' {ui.esc(cd["label"])}' + (f' {cd["seconds"]:.0f} s' if cd["seconds"] is not None else ""),
                 accent="purple", icon="🚥"),
        ]
        if em["mode"] != "NORMAL":
            cards.append(dict(label="Emergency vehicle", value="ACTIVE",
                              sub=ui.esc(f'{em["mode"]} · {em["target_phase"]}'), accent="red", icon="🚑", pulse=True))
        else:
            sub = "priority disabled" if not em["enabled"] else f'priority armed · {em["activations"]} activation(s)'
            cards.append(dict(label="Emergency vehicle", value="Clear", sub=sub, accent="green", icon="🚑"))
    in_use = snap is not None and snap["controller"] == "dqn"
    cards.append(dict(label="DQN model", value="Ready" if dq.get("tone") == "ok" else "Not trained",
                      sub=ui.esc(dq.get("detail", "")) + (" · <b>controlling signal</b>" if in_use else ""),
                      accent={"green": "green", "amber": "amber", "red": "red"}[dqn_tone], icon="🧠"))
    ui.kpi_grid(cards, cols=4)
    if snap is not None and snap["state"] == "FINISHED":
        ui.alert("The run finished. Detailed measured results are in the final summary; compare controllers on Page 6.", "green")


def _charts(sess) -> None:
    df = sess.series_df()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.traffic_chart(df), use_container_width=True, config=charts.PLOT_CONFIG, key="d_traffic")
    with c2:
        st.plotly_chart(charts.approach_queue_chart(df), use_container_width=True, config=charts.PLOT_CONFIG, key="d_queue")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.waiting_chart(df), use_container_width=True, config=charts.PLOT_CONFIG, key="d_wait")
    with c4:
        st.plotly_chart(charts.throughput_chart(df), use_container_width=True, config=charts.PLOT_CONFIG, key="d_thr")


def render() -> None:
    ui.page_header("Control Center", "Adaptive traffic-signal optimisation with computer vision and deep reinforcement learning",
                   "📊")
    sess = get_session()
    _overview()
    polling = sess.state == SessionState.RUNNING

    @st.fragment(run_every=1.0 if polling else None)
    def live() -> None:
        snap = sess.snapshot()
        if polling and (snap is None or snap["state"] != "RUNNING"):
            st.rerun()
        status = {s["key"]: s for s in system_status()}
        if snap is not None and snap.get("error"):
            ui.alert(snap["error"], "red", "Simulation error", preformatted=True)
        _kpis(snap, status)
        if snap is not None:
            st.progress(float(snap["progress"]), text=f'Simulated {snap["time"]:.0f} of {snap["duration"]} s')
        _charts(sess)

    live()