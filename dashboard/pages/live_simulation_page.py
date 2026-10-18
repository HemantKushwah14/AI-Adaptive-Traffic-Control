"""Page 2 - Live Traffic Simulation. The figure is drawn from the real TraCI snapshot."""
from __future__ import annotations

import streamlit as st

from config.constants import APPROACH_NAMES, APPROACHES, MOVEMENTS
from dashboard.components import ui
from dashboard.session import SessionState
from dashboard.sim_settings import CONTROLLERS, SPEED_OPTIONS
from dashboard.state import get_session, get_settings
from dashboard.visualizations import charts
from dashboard.visualizations.intersection import build_intersection_figure

MODE_TONE = {"GREEN": "green", "YELLOW": "amber", "ALL_RED": "red"}
STATE_TONE = {"IDLE": "blue", "RUNNING": "green", "PAUSED": "amber", "FINISHED": "cyan", "STOPPED": "amber", "ERROR": "red"}


# ------------------------------------------------------------------ side-panel cards
def _signal_card(snap: dict) -> str:
    sg, cd = snap["signal"], snap["countdown"]
    names = sg["phase_names"]
    transitioning = sg["mode"] != "GREEN"
    secs = "—" if cd["seconds"] is None else f'{cd["seconds"]:.0f}'
    greens = ", ".join(APPROACH_NAMES[a] for a in sg["green_approaches"]) or "none (clearance)"
    nxt = sg["target_phase"]
    body = (f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div><div class="big-phase">{ui.esc(names[sg["active_phase"]])}</div>'
            f'<div style="margin-top:6px">{ui.pill(sg["mode"].replace("_", "-"), MODE_TONE[sg["mode"]])}</div></div>'
            f'<div style="text-align:right"><div class="big-count">{secs}<span style="font-size:18px">s</span></div>'
            f'<div class="muted">{ui.esc(cd["label"])}</div></div></div>'
            f'<div class="muted" style="margin-top:10px">Green now: <b style="color:#22e58a">{ui.esc(greens)}</b>'
            + (f' · {ui.esc(cd["extra"])}' if cd.get("extra") else "") + '</div>'
            + ui.phase_chips(names, sg["active_phase"], nxt, transitioning)
            + f'<div class="muted" style="margin-top:8px">Next planned phase: <b>{ui.esc(names[nxt])}</b> · '
              f'phase switches so far: {sg["phase_switches"]}</div>')
    return ui.card("Traffic signal", body)


def _approach_card(snap: dict) -> str:
    rows = []
    for a in APPROACHES:
        i = snap["approaches"][a]
        dot = "ok" if i["green"] else "bad"
        rows.append((f'<span class="dot {dot}"></span>{APPROACH_NAMES[a]}', f'{i["count"]:.0f}', f'{i["queue"]:.0f}',
                     f'{i["waiting"]:.0f} s', f'{i["density"] * 100:.0f}%'))
    return ui.card("Approaches (live from TraCI)",
                   ui.html_table(["Approach", "Vehicles", "Queue", "Σ waiting", "Density"], rows))


def _emergency_card(snap: dict) -> str:
    em = snap["emergency"]
    if em["mode"] == "NORMAL" and not em["active_requests"]:
        head = ui.pill("No emergency vehicle detected", "green")
        extra = ""
    else:
        prog = em["clearance_progress"]
        head = ui.pill(f'PRIORITY: {em["mode"]}', "red")
        extra = (f'<div class="muted" style="margin-top:8px">Target: <b>{ui.esc(em["target_phase"])}</b> '
                 f'({ui.esc(em["target_approach"])}) · signal: {ui.esc(em["signal_transition"])}'
                 + (f' · clearance {prog * 100:.0f}%' if prog is not None else "") + '</div>')
    log = "".join(f'<div class="logline">t={m["time"]:.0f}s · {ui.esc(m["message"])}</div>' for m in em["log"][-4:])
    note = "" if em["enabled"] else '<div class="muted">Priority is <b>disabled</b>: ambulances are only observed.</div>'
    src = '<div class="muted" style="margin-top:6px">Detection source: simulation ground truth (SIMULATED)</div>'
    return ui.card("Emergency priority", head + extra + note + src + (f'<div style="margin-top:8px">{log}</div>' if log else ""))


def _controller_card(snap: dict) -> str:
    if snap["controller"] == "dqn":
        d = snap["dqn"]
        last = d["last"] if d else None
        if not last:
            return ui.card("DQN decision", '<div class="muted">Waiting for the first decision…</div>')
        q = " · ".join(f'{n}: {v:.2f}' for n, v in zip(snap["signal"]["phase_names"], last["q_values"]))
        body = (f'<div style="font-weight:700;color:#fff">{ui.esc(last["explanation"]["summary"])}</div>'
                f'<div class="muted" style="margin-top:6px">Q-values: {ui.esc(q)}</div>'
                f'<div class="muted" style="margin-top:6px">{d["stats"]["decisions"]} decisions · {d["stats"]["switches"]} switches · '
                f'{d["stats"]["overrides"]} safety overrides · ε = 0 (greedy)</div>')
        return ui.card("DQN decision", body)
    return ui.card("Fixed-time plan", '<div class="muted">Constant green durations for NS-through, NS-left, EW-through, EW-left '
                                      '(35 / 10 / 35 / 10 s by default). No learning is involved.</div>')


def _stats_cards(snap: dict) -> None:
    k = snap["kpi"]
    ui.kpi_grid([
        dict(label="Vehicles", value=ui.fmt(k["vehicles_in_network"], 0), sub=f'{k["vehicles_arrived"]} arrived', accent="cyan", icon="🚗"),
        dict(label="Avg waiting", value=ui.fmt(k["avg_waiting_time_s"], 1, " s"), sub="per vehicle", accent="blue", icon="⏱"),
        dict(label="Avg queue", value=ui.fmt(k["avg_queue_length"], 1), sub=f'now {k["queue_now"]:.0f}', accent="amber", icon="🚧"),
        dict(label="Throughput", value=ui.fmt(k["throughput_now_vph"], 0), sub="veh/h, trailing 60 s", accent="green", icon="🏁"),
    ], cols=2)


# ------------------------------------------------------------------ page
def render() -> None:
    ui.page_header("Live Traffic Simulation",
                   "Top-down view of the real SUMO intersection. Vehicles and signals are streamed from TraCI.", "🚦")
    sess, settings = get_session(), get_settings()
    active = sess.is_active

    # ---- controls ----
    with st.container(border=True):
        c = st.columns([1, 1, 1, 1, 1.5, 1.7, 1.2])
        start = c[0].button("▶  Start", type="primary", disabled=active, use_container_width=True)
        pause_label = "⏯  Resume" if sess.state == SessionState.PAUSED else "⏸  Pause"
        toggle = c[1].button(pause_label, disabled=not active, use_container_width=True)
        stop = c[2].button("⏹  Stop", disabled=not active, use_container_width=True)
        reset = c[3].button("↺  Reset", use_container_width=True)
        cur_speed = next((k for k, v in SPEED_OPTIONS.items() if v == settings["speed"]), "5×")
        speed_label = c[4].select_slider("Simulation speed", options=list(SPEED_OPTIONS), value=cur_speed)
        ctrl_keys = list(CONTROLLERS)
        ctrl_choice = c[5].selectbox("Controller", ctrl_keys, index=ctrl_keys.index(settings["controller"]),
                                     format_func=lambda k: CONTROLLERS[k], disabled=active)
        priority = c[6].toggle("Emergency priority", value=bool(settings["emergency_priority"]), disabled=active)
        with st.expander("Display and run options"):
            o = st.columns(3)
            view_range = o[0].slider("View half-width (m)", 60, 200, 110, 10)
            scale = o[1].slider("Vehicle drawing scale", 1.0, 2.0, 1.2, 0.1)
            step = o[2].selectbox("Time step", [1.0, 0.5], index=0 if settings["step_length"] == 1.0 else 1,
                                  format_func=lambda v: f"{v:.1f} s" + ("  (matches evaluation)" if v == 1.0 else "  (smoother motion)"),
                                  disabled=active)
            st.caption(f'Scenario: {settings["demand_vph"]["N"]:.0f}/{settings["demand_vph"]["S"]:.0f}/'
                       f'{settings["demand_vph"]["E"]:.0f}/{settings["demand_vph"]["W"]:.0f} veh/h (N/S/E/W) · '
                       f'{settings["duration"]} s · seed {settings["seed"]} · green {settings["min_green"]}-{settings["max_green"]} s. '
                       'Change these on Page 7. Vehicles are drawn larger than real size for visibility.')

    settings.update(speed=SPEED_OPTIONS[speed_label])
    sess.set_speed(settings["speed"])
    if not active:
        settings.update(controller=ctrl_choice, emergency_priority=priority, step_length=step)

    # ---- actions ----
    if start:
        with st.spinner("Starting SUMO…"):
            sess.start(settings)
        active = sess.is_active
    elif toggle:
        sess.resume() if sess.state == SessionState.PAUSED else sess.pause()
    elif stop:
        sess.stop()
    elif reset:
        sess.reset()
    if start or toggle or stop or reset:
        st.rerun()

    snap0 = sess.snapshot()
    if sess.state == SessionState.ERROR and sess.error:
        ui.sumo_error(sess.error)
    for w in sess.warnings:
        ui.alert(w, "amber", "Compatibility notice")

    # ---- ambulance dispatch ----
    with st.expander("🚑  Dispatch a simulated ambulance (demo)"):
        d = st.columns([1.2, 1.2, 1.6, 2])
        ap = d[0].selectbox("Approach", APPROACHES, format_func=lambda a: APPROACH_NAMES[a])
        mv = d[1].selectbox("Movement", MOVEMENTS, index=1)
        if d[2].button("Dispatch ambulance", disabled=not sess.is_active, use_container_width=True):
            try:
                vid = sess.dispatch_ambulance(ap, mv)
                st.success(f"{vid} injected into SUMO (simulated vehicle).")
            except Exception as exc:
                ui.alert(str(exc), "red", "Could not dispatch")
        d[3].caption("The vehicle is real inside SUMO, but its detection is simulated ground truth, not computer vision.")

    # ---- live region ----
    polling = sess.state == SessionState.RUNNING

    @st.fragment(run_every=0.5 if polling else None)
    def live() -> None:
        snap = sess.snapshot()
        if polling and (snap is None or snap["state"] != "RUNNING"):
            st.rerun()
        left, right = st.columns([1.7, 1])
        with left:
            try:
                fig = build_intersection_figure(snap, view_range, scale, height=660)
                st.plotly_chart(fig, use_container_width=True, config=charts.STATIC_CONFIG, key="intersection")
            except (FileNotFoundError, RuntimeError) as exc:
                ui.alert(str(exc), "red", "Cannot draw intersection", preformatted=True)
            if snap:
                tone = STATE_TONE[snap["state"]]
                rate = f' · {snap["sim_rate"]:.1f}× real time' if snap["state"] == "RUNNING" and snap["sim_rate"] else ""
                st.progress(float(snap["progress"]), text=f'{snap["state"].title()} · t = {snap["time"]:.0f} / {snap["duration"]} s{rate}')
                if snap["state"] == "FINISHED" and snap.get("summary"):
                    s = snap["summary"]
                    ui.alert(f'Run finished (measured): avg waiting {s["avg_waiting_time_s"]:.1f} s · avg queue '
                             f'{s["avg_queue_length"]:.1f} · throughput {s["throughput_veh_per_hour"]:.0f} veh/h · '
                             f'avg travel {s["avg_travel_time_s"]:.1f} s · {s["vehicles_arrived"]} of {s["vehicles_entered"]} vehicles arrived.',
                             "green", "Final summary")
        with right:
            if snap:
                ui.md(_signal_card(snap))
                _stats_cards(snap)
                ui.md(_approach_card(snap))
                ui.md(_emergency_card(snap))
                ui.md(_controller_card(snap))
            else:
                ui.md(ui.card("Ready", '<div class="muted">Choose a controller and press <b>Start</b>. SUMO is launched in the background, '
                                       'so page reruns never restart it.</div>'))
        df = sess.series_df()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.traffic_chart(df, 250), use_container_width=True, config=charts.PLOT_CONFIG, key="l_traffic")
        with c2:
            st.plotly_chart(charts.approach_queue_chart(df, 250), use_container_width=True, config=charts.PLOT_CONFIG, key="l_queue")
        st.plotly_chart(charts.phase_timeline(df), use_container_width=True, config=charts.PLOT_CONFIG, key="l_phase")

    live()