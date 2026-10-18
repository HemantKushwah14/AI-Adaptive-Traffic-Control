"""Page 7 - Settings & Experiments. Inputs are validated with the same SimulationConfig.validate()
the simulator uses, plus checkpoint / fixed-plan checks and an exact vehicle-count preview."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import streamlit as st

from config.constants import APPROACH_NAMES, APPROACHES
from config.rl_config import DQN_DIR
from config.settings import GENERATED_DIR, ensure_dirs
from dashboard.components import ui
from dashboard.jobs import format_events, parse_events
from dashboard.session import SessionState
from dashboard.sim_settings import CONTROLLERS, SPEED_OPTIONS, build_sim_config, default_settings
from dashboard.state import get_session, get_settings, save_settings
from reinforcement_learning.dqn_agent import CheckpointError
from simulation.adaptive_controller import AdaptiveController
from simulation.fixed_time_controller import FixedTimeController
from simulation.route_generator import generate_routes

PRESETS = {
    "Light (250 each)": {a: 250.0 for a in APPROACHES},
    "Balanced (500 each)": {a: 500.0 for a in APPROACHES},
    "N-S rush (800 / 300)": {"N": 800.0, "S": 800.0, "E": 300.0, "W": 300.0},
    "Heavy (900 each)": {a: 900.0 for a in APPROACHES},
}
AUTO = "Auto (best.pt, else last.pt)"


def _version() -> int:
    return int(st.session_state.get("settings_ver", 0))


def _bump() -> None:
    st.session_state["settings_ver"] = _version() + 1


def _apply_preset(demand: dict) -> None:
    get_settings()["demand_vph"] = dict(demand)
    _bump()


def _reset_defaults() -> None:
    save_settings(default_settings())
    _bump()


def _checkpoints() -> List[Path]:
    return sorted(DQN_DIR.glob("*.pt")) if DQN_DIR.exists() else []


def render() -> None:
    ui.page_header("Settings & Experiments", "Configure the scenario, signal limits, controller and emergency options. Applied at the next Start / run.", "⚙️")
    cur, sess, ver = get_settings(), get_session(), _version()
    if sess.state in (SessionState.RUNNING, SessionState.PAUSED):
        ui.alert("A simulation is running. New settings apply to the next Start; the running one is unchanged.", "blue")

    with st.container(border=True):
        st.markdown("**Experiment presets (traffic demand)**")
        cols = st.columns(len(PRESETS))
        for col, (name, dem) in zip(cols, PRESETS.items()):
            col.button(name, on_click=_apply_preset, args=(dem,), use_container_width=True, key=f"preset_{name}")
        st.caption("Presets only change demand. The DQN was trained on roughly 300-800 veh/h per approach; other values trigger a compatibility notice.")

    ckpts = _checkpoints()
    ck_options = [AUTO] + [p.name for p in ckpts]
    cur_ck = Path(cur["checkpoint"]).name if cur.get("checkpoint") else AUTO
    with st.form(f"settings_form_{ver}"):
        st.markdown("**Traffic demand (vehicles per hour, per approach)**")
        dc = st.columns(4)
        demand = {a: dc[i].number_input(APPROACH_NAMES[a], 0.0, 3000.0, float(cur["demand_vph"][a]), 50.0,
                                        key=f"set_dem_{a}_{ver}") for i, a in enumerate(APPROACHES)}
        c = st.columns(4)
        duration = c[0].number_input("Simulation duration (s)", 60, 7200, int(cur["duration"]), 60, key=f"set_dur_{ver}")
        seed = c[1].number_input("Random seed", 0, 1_000_000, int(cur["seed"]), key=f"set_seed_{ver}")
        limit_on = c[2].checkbox("Limit total vehicles", value=cur.get("max_vehicles") is not None, key=f"set_lim_{ver}")
        max_veh = c[3].number_input("Max vehicles", 1, 20000, int(cur.get("max_vehicles") or 500), 50, key=f"set_maxv_{ver}")
        st.markdown("**Signal timing and emergency**")
        s = st.columns(4)
        min_g = s[0].number_input("Minimum green (s)", 3, 60, int(cur["min_green"]), key=f"set_ming_{ver}")
        max_g = s[1].number_input("Maximum green (s)", 5, 180, int(cur["max_green"]), key=f"set_maxg_{ver}")
        emp = s[2].slider("Emergency vehicle probability", 0.0, 0.2, float(cur["emergency_probability"]), 0.01, format="%.2f",
                          key=f"set_emp_{ver}")
        speed_label = s[3].select_slider("Simulation speed", list(SPEED_OPTIONS),
                                         value=next((k for k, v in SPEED_OPTIONS.items() if v == cur["speed"]), "5×"),
                                         key=f"set_speed_{ver}")
        ev_text = st.text_input("Scheduled simulated ambulances (time:approach:movement, comma-separated; optional)",
                                value=format_events(cur.get("emergency_events", [])), key=f"set_ev_{ver}")
        st.markdown("**Controller**")
        k = st.columns(3)
        ctrl_keys = list(CONTROLLERS)
        controller = k[0].selectbox("Controller", ctrl_keys, index=ctrl_keys.index(cur["controller"]),
                                    format_func=lambda x: CONTROLLERS[x], key=f"set_ctrl_{ver}")
        ck_choice = k[1].selectbox("DQN checkpoint", ck_options, index=ck_options.index(cur_ck) if cur_ck in ck_options else 0,
                                   key=f"set_ck_{ver}")
        priority = k[2].toggle("Emergency priority", value=bool(cur["emergency_priority"]), key=f"set_prio_{ver}")
        g = st.columns(4)
        greens = tuple(g[i].number_input(f"Fixed green: {n} (s)", 3.0, 120.0, float(cur["fixed_greens"][i]), 1.0, key=f"set_fg{i}_{ver}")
                       for i, n in enumerate(["NS through", "NS left", "EW through", "EW left"]))
        submitted = st.form_submit_button("✔  Validate and apply", type="primary")

    b = st.columns([1, 1, 4])
    b[0].button("↺  Reset to defaults", on_click=_reset_defaults, use_container_width=True)
    b[1].download_button("⬇  Settings (JSON)", json.dumps(cur, indent=2, default=str).encode("utf-8"),
                         file_name="settings.json", mime="application/json", use_container_width=True)

    if submitted:
        new = dict(cur)
        try:
            new.update(demand_vph=demand, duration=int(duration), seed=int(seed),
                       max_vehicles=int(max_veh) if limit_on else None, min_green=int(min_g), max_green=int(max_g),
                       emergency_probability=float(emp), speed=SPEED_OPTIONS[speed_label],
                       emergency_events=parse_events(ev_text) if ev_text.strip() else [], controller=controller,
                       checkpoint=None if ck_choice == AUTO else str(DQN_DIR / ck_choice), emergency_priority=bool(priority),
                       fixed_greens=greens)
            cfg = build_sim_config(new)               # SimulationConfig.validate()
            FixedTimeController(greens)               # validates the fixed plan
            notices: List[str] = []
            if controller == "dqn":
                ctrl = AdaptiveController.from_checkpoint(Path(new["checkpoint"]) if new["checkpoint"] else None)
                notices = ctrl.compatibility_warnings(cfg)
            ensure_dirs()
            preview = generate_routes(cfg, GENERATED_DIR / "preview_routes.rou.xml")
        except (ValueError, CheckpointError, FileNotFoundError, RuntimeError) as exc:
            ui.alert(str(exc), "red", "Settings not applied", preformatted=True)
        else:
            save_settings(new)
            ui.alert(f'Settings applied. This scenario generates {preview.total_vehicles} vehicles '
                     f'({", ".join(f"{a}: {n}" for a, n in preview.per_approach.items())}), including '
                     f'{len(preview.ambulances)} simulated ambulance(s). Decision interval = {cfg.decision_interval} s.', "green", "Valid")
            if cfg.decision_interval != 5 and controller == "dqn":
                notices.append("Decision interval differs from the 5 s used in DQN training.")
            for n in notices:
                ui.alert(n, "amber", "Compatibility notice")
    with st.expander("Current settings"):
        st.json(json.loads(json.dumps(cur, default=str)))