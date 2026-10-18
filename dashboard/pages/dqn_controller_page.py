"""Page 4 - DQN Controller: live decision, Q-values, state, training curves, checkpoint info,
and a what-if probe on a SYNTHETIC state. Q-values are not a causal explanation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.constants import (APPROACH_NAMES, APPROACHES, DENSITY_CAPACITY_VEH, NUM_PHASES, PHASE_NAMES,
                              QUEUE_NORM_VEH, STATE_DIM, WAITING_NORM_S)
from config.rl_config import TRAIN_LOG, TRAIN_SUMMARY
from dashboard.components import ui
from dashboard.session import SessionState
from dashboard.state import get_session, get_settings
from dashboard.styles.theme import COLORS
from dashboard.visualizations import charts
from reinforcement_learning.dqn_agent import CheckpointError, DQNAgent, read_checkpoint_meta
from simulation.adaptive_controller import CAVEAT, default_checkpoint_path

PHASE_HEX = charts.PHASE_COLOR


def _ckpt_path() -> Path:
    ck = get_settings().get("checkpoint")
    p = Path(ck) if ck else default_checkpoint_path()
    if not p.exists():
        raise CheckpointError(f"Checkpoint not found: {p}")
    return p


@st.cache_data(show_spinner=False)
def _meta(path: str, mtime: float) -> dict:
    return read_checkpoint_meta(Path(path))


@st.cache_resource(show_spinner=False)
def _agent(path: str, mtime: float) -> DQNAgent:
    return DQNAgent.from_checkpoint(Path(path), with_buffer=False)


def _q_chart(q: List[float], mask: List[bool], applied: Optional[int], height: int = 280) -> go.Figure:
    colors = [COLORS["green"] if i == applied else (COLORS["blue"] if mask[i] else "#4a5578") for i in range(NUM_PHASES)]
    fig = go.Figure(go.Bar(x=list(PHASE_NAMES), y=q, marker_color=colors, text=[f"{v:.2f}" for v in q],
                           textposition="outside"))
    fig = charts.style(fig, height, "Q-values per phase (green = applied · grey = illegal right now)", legend=False)
    fig.update_xaxes(title_text="")
    return fig


def _line(df: pd.DataFrame, x: str, ys: List[Tuple[str, str, str]], title: str, height: int = 250,
          xtitle: str = "episode") -> go.Figure:
    fig = go.Figure()
    for col, name, color in ys:
        if col in df:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], name=name, mode="lines", line=dict(color=color, width=2)))
    fig = charts.style(fig, height, title)
    fig.update_xaxes(title_text=xtitle)
    return fig


def _state_table(state: List[float], elapsed: float) -> str:
    rows = []
    for i, a in enumerate(APPROACHES):
        d, q, w = state[i], state[4 + i], state[8 + i]
        rows.append((APPROACH_NAMES[a], f"{d:.2f} (≈{d * DENSITY_CAPACITY_VEH:.0f} veh)",
                     f"{q:.2f} (≈{q * QUEUE_NORM_VEH:.0f} halting)", f"{w:.2f} (≈{w * WAITING_NORM_S:.0f} s)"))
    ph = int(np.argmax(state[12:16]))
    table = ui.html_table(["Approach", "Density [0,1]", "Queue [0,1]", "Waiting [0,1]"], rows)
    return table + (f'<div class="muted" style="margin-top:8px">Phase one-hot → <b>{PHASE_NAMES[ph]}</b> · '
                    f'elapsed green {elapsed:.0f} s (state value {state[16]:.2f} of max green)</div>')


# ------------------------------------------------------------------ live section
def _live_content(sess) -> None:
    snap = sess.snapshot()
    if snap is None or snap["controller"] != "dqn" or not snap.get("dqn"):
        ui.alert("No DQN run is active. Open Live Traffic Simulation, choose the DQN controller and press Start to see "
                 "live decisions here. You can still inspect the checkpoint, the training history and the what-if probe below.",
                 "blue", "Live decision view")
        return
    dq = snap["dqn"]
    last = dq["last"]
    sg = snap["signal"]
    info = dq["info"]
    if not last:
        ui.alert("Waiting for the first decision ...", "blue")
        return
    names = list(PHASE_NAMES)
    applied, requested = last["applied"], last["requested"]
    hold = applied == last["active_before"]
    sub = ui.pill("hold" if hold else "switch", "blue" if hold else "amber")
    if last["override_reason"]:
        sub += " " + ui.pill(f'safety override: {last["override_reason"]}', "red")
    ui.kpi_grid([
        dict(label="DQN-selected action", value=ui.esc(names[applied]), sub=sub, accent="cyan", icon="🧠"),
        dict(label="Current signal phase", value=ui.esc(names[sg["active_phase"]]),
             sub=ui.pill(sg["mode"].replace("_", "-"), {"GREEN": "green", "YELLOW": "amber", "ALL_RED": "red"}[sg["mode"]]),
             accent="green", icon="🚥"),
        dict(label="Next planned phase", value=ui.esc(names[sg["target_phase"]]),
             sub="after yellow + all-red" if sg["target_phase"] != sg["active_phase"] else "same as current", accent="purple", icon="⏭"),
        dict(label="Epsilon (exploration)", value="0.00",
             sub=f'greedy inference · ε at end of training: {info["checkpoint_epsilon"]:.2f}', accent="amber", icon="🎲"),
        dict(label="Decisions", value=str(dq["stats"]["decisions"]),
             sub=f'{dq["stats"]["switches"]} switches · {dq["stats"]["overrides"]} safety overrides', accent="blue", icon="🔢"),
        dict(label="Checkpoint", value=f'{info["episodes_trained"]} ep',
             sub=ui.esc(Path(info["checkpoint"]).name if info["checkpoint"] else "?"), accent="green", icon="💾"),
    ], cols=3)
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(_q_chart(last["q_values"], last["mask"], applied), use_container_width=True,
                        config=charts.PLOT_CONFIG, key="dqn_q")
    with right:
        ui.md(ui.card("Current traffic state (input to the network, normalised)",
                      _state_table(last["state"], last["elapsed_green"])))
    ex = last["explanation"]
    rank = "".join(f'<div class="logline">{"✓ " if r["legal"] else "✗ "}{ui.esc(r["name"])}: Q = {r["q"]:.2f}'
                   f'{"" if r["legal"] else " (illegal now)"}</div>' for r in ex["ranking"])
    press = " · ".join(f'{ui.esc(k)} {v:.0f}' for k, v in ex["phase_pressure"].items())
    agree = "agrees" if ex["agrees_with_queue_pressure"] else "differs from"
    body = ("".join(f'<div class="logline">{ui.esc(l)}</div>' for l in ex["lines"])
            + f'<div class="muted" style="margin-top:8px">Queue pressure (vehicles waiting on the phase\'s approaches): {press}. '
              f'The chosen phase {agree} the highest-queue-pressure phase.</div>'
            + f'<div style="margin-top:8px">{rank}</div>'
            + f'<div class="muted" style="margin-top:8px"><i>{ui.esc(CAVEAT)}</i></div>')
    ui.md(ui.card("Why this decision? (from the state and Q-values)", body))
    hist = dq["history"]
    if hist:
        hdf = pd.DataFrame({"time": [h["time"] for h in hist]})
        for i, n in enumerate(PHASE_NAMES):
            hdf[n] = [h["q"][i] for h in hist]
        fig = _line(hdf, "time", [(n, n, PHASE_HEX[i]) for i, n in enumerate(PHASE_NAMES)],
                    "Q-values over the run (one point per decision)", 260, "simulation time (s)")
        st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG, key="dqn_qhist")


# ------------------------------------------------------------------ checkpoint + training
def _checkpoint_section() -> Optional[Path]:
    try:
        path = _ckpt_path()
    except CheckpointError as exc:
        ui.alert(str(exc), "amber", "No DQN checkpoint available")
        return None
    try:
        m = _meta(str(path), path.stat().st_mtime)
    except CheckpointError as exc:
        ui.alert(str(exc), "red", "Checkpoint unreadable")
        return None
    cfg, meta = m["config"], m["meta"]
    rows = [("File", ui.esc(path.name)), ("Size", f'{m["size_kb"]:.0f} KB'),
            ("Saved", datetime.fromtimestamp(m["modified"]).strftime("%Y-%m-%d %H:%M")),
            ("Training episodes", str(m["episodes"])), ("Stored transitions / learning steps", f'{m["env_steps"]} / {m["learn_steps"]}'),
            ("Network", " → ".join(str(x) for x in [cfg["state_dim"], *cfg["hidden"], cfg["action_dim"]])),
            ("γ · lr · batch · double-DQN", f'{cfg["gamma"]} · {cfg["lr"]} · {cfg["batch_size"]} · {cfg["double_dqn"]}'),
            ("Best validation waiting time", "—" if meta.get("best_val_wait_s") in (None, float("inf")) else f'{meta["best_val_wait_s"]:.1f} s'),
            ("Training demand range (veh/h)", str(meta.get("demand_range", "—"))),
            ("Reward weights", ui.esc(str(meta.get("reward_weights", "—"))))]
    ui.md(ui.card("Model checkpoint", ui.html_table(["Property", "Value"], rows)))
    return path


def _training_section() -> None:
    if not TRAIN_LOG.exists():
        ui.alert("No training log yet. Train the agent (Part 4): python -m reinforcement_learning.train --episodes 80",
                 "amber", "Training curves")
        return
    try:
        df = pd.read_csv(TRAIN_LOG)
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        ui.alert(str(exc), "red", "Training log unreadable")
        return
    if df.empty:
        ui.alert("The training log is empty.", "amber")
        return
    for col in ("total_reward", "mean_loss", "epsilon", "avg_wait_s", "val_dqn_wait_s", "val_fixed_wait_s"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if len(df) >= 5:
        df["reward_ma5"] = df["total_reward"].rolling(5).mean()
    last = df.iloc[-1]
    ui.kpi_grid([
        dict(label="Episodes logged", value=str(int(last["episode"])), sub="in training_log.csv", accent="blue", icon="📚"),
        dict(label="Last training reward", value=ui.fmt(last["total_reward"], 1), sub="episode total (demand varies per episode)", accent="green", icon="🏆"),
        dict(label="Last training loss", value=ui.fmt(last["mean_loss"], 4), sub="mean Huber loss", accent="amber", icon="📉"),
        dict(label="Epsilon at last episode", value=ui.fmt(last["epsilon"], 2), sub="exploration rate", accent="cyan", icon="🎲")], cols=4)
    c1, c2 = st.columns(2)
    c1.plotly_chart(_line(df, "episode", [("total_reward", "Episode reward", COLORS["cyan"]),
                                          ("reward_ma5", "Moving avg (5)", COLORS["amber"])], "Training reward"),
                    use_container_width=True, config=charts.PLOT_CONFIG, key="tr_reward")
    c2.plotly_chart(_line(df.dropna(subset=["mean_loss"]), "episode", [("mean_loss", "Mean loss", COLORS["red"])],
                          "Training loss"), use_container_width=True, config=charts.PLOT_CONFIG, key="tr_loss")
    c3, c4 = st.columns(2)
    c3.plotly_chart(_line(df, "episode", [("epsilon", "Epsilon", COLORS["purple"])], "Exploration rate (epsilon)"),
                    use_container_width=True, config=charts.PLOT_CONFIG, key="tr_eps")
    val = df.dropna(subset=["val_dqn_wait_s"]) if "val_dqn_wait_s" in df else pd.DataFrame()
    if len(val):
        c4.plotly_chart(_line(val, "episode", [("val_dqn_wait_s", "DQN (validation seeds)", COLORS["cyan"]),
                                               ("val_fixed_wait_s", "Fixed-time (validation seeds)", COLORS["amber"])],
                              "Validation: average waiting time (s)"), use_container_width=True,
                        config=charts.PLOT_CONFIG, key="tr_val")
    else:
        c4.markdown('<div class="muted">No validation points yet (validation runs every --val-every episodes).</div>',
                    unsafe_allow_html=True)
    if TRAIN_SUMMARY.exists():
        st.caption(f"Training summary file: {TRAIN_SUMMARY.name} - validation numbers only; held-out test results are on Page 6.")


# ------------------------------------------------------------------ what-if probe
def _probe(path: Path) -> None:
    s = get_settings()
    min_g, max_g = int(s["min_green"]), int(s["max_green"])
    with st.expander("🔬  What-if probe: ask the network about a SYNTHETIC traffic state"):
        st.caption("This does not use the running simulation. You define a hypothetical state; the checkpoint answers. "
                   "It shows what the network prefers, not what is optimal.")
        cols = st.columns(4)
        cnt, hal, wai = {}, {}, {}
        for i, a in enumerate(APPROACHES):
            with cols[i]:
                st.markdown(f"**{APPROACH_NAMES[a]}**")
                cnt[a] = st.slider("Vehicles on approach", 0, 40, 10, key=f"pr_c_{a}")
                hal[a] = st.slider("Halting (queue)", 0, 30, 5, key=f"pr_q_{a}")
                wai[a] = st.slider("Σ waiting time (s)", 0, 1000, 100, 25, key=f"pr_w_{a}")
        c = st.columns(2)
        active = c[0].selectbox("Active phase", list(range(NUM_PHASES)), format_func=lambda p: PHASE_NAMES[p], key="pr_ph")
        elapsed = c[1].slider("Elapsed green (s)", 0, max_g, min(15, max_g), key="pr_el")
        state = np.asarray([min(cnt[a] / DENSITY_CAPACITY_VEH, 1.0) for a in APPROACHES]
                           + [min(hal[a] / QUEUE_NORM_VEH, 1.0) for a in APPROACHES]
                           + [min(wai[a] / WAITING_NORM_S, 1.0) for a in APPROACHES]
                           + [1.0 if p == active else 0.0 for p in range(NUM_PHASES)]
                           + [min(elapsed / max_g, 1.0)], dtype=np.float32)
        assert state.shape == (STATE_DIM,)
        mask = np.ones(NUM_PHASES, dtype=bool)                  # same rule as reinforcement_learning.action_mask
        if elapsed < min_g:
            mask[:] = False
            mask[active] = True
        elif elapsed + min(5, min_g) > max_g:
            mask[active] = False
        try:
            agent = _agent(str(path), path.stat().st_mtime)
            q = agent.q_values(state)
            choice = agent.select_action(state, mask, greedy=True)
        except (CheckpointError, ValueError) as exc:
            ui.alert(str(exc), "red", "Probe failed")
            return
        st.plotly_chart(_q_chart([float(x) for x in q], [bool(m) for m in mask], choice, 260),
                        use_container_width=True, config=charts.PLOT_CONFIG, key="probe_q")
        legal = [PHASE_NAMES[i] for i in range(NUM_PHASES) if mask[i]]
        st.markdown(f"Greedy choice among legal phases ({', '.join(legal)}): **{PHASE_NAMES[choice]}** "
                    f"({'hold' if choice == active else 'switch'}).")
        st.caption(CAVEAT)


def render() -> None:
    ui.page_header("DQN Controller", "State, Q-values, decisions, training history and checkpoint of the traffic-signal agent.", "🧠")
    sess = get_session()
    polling = sess.state == SessionState.RUNNING

    @st.fragment(run_every=1.0 if polling else None)
    def live() -> None:
        if polling and sess.state != SessionState.RUNNING:
            st.rerun()
        _live_content(sess)

    live()
    st.markdown("### Model and training")
    path = _checkpoint_section()
    _training_section()
    if path is not None:
        _probe(path)