"""Page 6 - Performance Comparison: fixed-time vs DQN on identical demand/duration/seeds.
Runs in a background job; results are read from the saved report (real measurements only)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.rl_config import TEST_SEED_BASE
from dashboard.components import ui
from dashboard.jobs import JobState, comparison_target, get_comparison_job
from dashboard.sim_settings import build_sim_config
from dashboard.state import get_settings
from dashboard.styles.theme import COLORS
from dashboard.visualizations import charts
from evaluation.reports import csv_bytes, load_latest
from reinforcement_learning.dqn_agent import CheckpointError
from simulation.adaptive_controller import default_checkpoint_path

LABEL_NAME = {"fixed_time": ("Fixed-time", COLORS["amber"]), "dqn": ("DQN", COLORS["cyan"])}


def _job_progress(job) -> None:
    running = job.snapshot()["state"] == JobState.RUNNING.value

    @st.fragment(run_every=1.0 if running else None)
    def view() -> None:
        s = job.snapshot()
        if running and s["state"] != JobState.RUNNING.value:
            st.rerun()
        if s["state"] == JobState.RUNNING.value:
            st.progress(min(s["done"] / max(s["total"], 1), 1.0),
                        text=f'{s["message"]} · {s["done"]}/{s["total"]} runs · {s["elapsed_s"]:.0f} s elapsed')
            if st.button("Cancel (after current run)", key="cmp_cancel"):
                job.cancel()
        elif s["state"] == JobState.ERROR.value:
            ui.alert(s["error"], "red", "Comparison failed", preformatted=True)
        elif s["state"] == JobState.CANCELLED.value:
            ui.alert("The comparison was cancelled; the previous saved report (if any) is shown below.", "amber")

    view()


def _controls(settings: dict) -> None:
    job = get_comparison_job()
    with st.container(border=True):
        c = st.columns([1, 1, 3])
        n_seeds = c[0].slider("Number of seeds", 1, 10, 3, key="cmp_n")
        base_seed = c[1].number_input("First seed", 0, 1_000_000, TEST_SEED_BASE, key="cmp_seed")
        d = settings["demand_vph"]
        c[2].caption(f'Uses the settings from Page 7: demand N/S/E/W {d["N"]:.0f}/{d["S"]:.0f}/{d["E"]:.0f}/{d["W"]:.0f} veh/h · '
                     f'{settings["duration"]} s · green {settings["min_green"]}-{settings["max_green"]} s · fixed plan {settings["fixed_greens"]} s · '
                     f'checkpoint {settings["checkpoint"] or "auto (best.pt, else last.pt)"}. Always 1.0 s steps and no ambulances. '
                     f'Seeds ≥ {TEST_SEED_BASE} are held out from DQN training/validation.')
        if int(base_seed) < TEST_SEED_BASE:
            ui.alert(f"Seeds below {TEST_SEED_BASE} may overlap training or validation seeds, so the result is not strictly held-out.", "amber")
        running = job.snapshot()["state"] == JobState.RUNNING.value
        if st.button(f"▶  Run comparison ({2 * n_seeds} simulations)", type="primary", disabled=running, key="cmp_run"):
            try:
                cfg = dataclasses.replace(build_sim_config(settings), step_length=1.0, emergency_probability=0.0,
                                          emergency_events=[])
                cfg.validate()
                ck = settings.get("checkpoint")
                ckpt = Path(ck) if ck else default_checkpoint_path()
                if not ckpt.exists():
                    raise CheckpointError(f"Checkpoint not found: {ckpt}")
                seeds = [int(base_seed) + i for i in range(n_seeds)]
                job.start(comparison_target, cfg, ckpt, seeds, tuple(settings["fixed_greens"]))
                st.rerun()
            except (ValueError, CheckpointError) as exc:
                ui.alert(str(exc), "red", "Cannot start the comparison", preformatted=True)
    _job_progress(job)


def _improvement_bar(imp) -> go.Figure:
    vals = imp["improvement_%"].astype(float)
    colors = [COLORS["green"] if (v == v and v > 0) else COLORS["red"] for v in vals]
    text = [f"{v:+.1f}%" if v == v else "n/a" for v in vals]
    fig = go.Figure(go.Bar(x=vals.fillna(0), y=list(imp.index), orientation="h", marker_color=colors, text=text,
                           textposition="outside"))
    fig = charts.style(fig, 300, "DQN improvement over fixed-time (positive = DQN better)", legend=False)
    fig.update_xaxes(title_text="improvement (%)", zeroline=True, zerolinecolor=COLORS["muted"])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return fig


def _metric_bars(imp) -> go.Figure:
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(imp.index), vertical_spacing=0.2)
    for i, (label, row) in enumerate(imp.iterrows()):
        r, c = divmod(i, 3)
        std = np.nan_to_num([row["fixed_time_std"], row["dqn_std"]])
        fig.add_trace(go.Bar(x=["Fixed-time", "DQN"], y=[row["fixed_time"], row["dqn"]],
                             error_y=dict(type="data", array=list(std), visible=True, color=COLORS["muted"]),
                             marker_color=[COLORS["amber"], COLORS["cyan"]], showlegend=False), row=r + 1, col=c + 1)
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=40, r=14, t=40, b=30), font=dict(color=COLORS["muted"], size=12))
    fig.update_annotations(font_size=12, font_color=COLORS["text"])
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


def _ts_line(ts, column: str, title: str) -> go.Figure:
    g = ts.groupby(["controller", "time"])[column].mean().unstack("controller")
    step = max(1, len(g) // 600)
    g = g.iloc[::step]
    fig = go.Figure()
    for key, (name, color) in LABEL_NAME.items():
        if key in g:
            fig.add_trace(go.Scatter(x=g.index, y=g[key], name=name, mode="lines", line=dict(color=color, width=2)))
    return charts.style(fig, 290, title)


def _results() -> None:
    try:
        data = load_latest()
    except ValueError as exc:
        ui.alert(str(exc), "red", "Saved comparison unreadable")
        return
    if data is None:
        ui.alert("No comparison has been run yet. Press Run above (needs a trained DQN checkpoint), or: "
                 "python scripts\\run_comparison.py --seeds 3", "blue", "No results yet")
        return
    rep, imp = data["report"], data["improvement"]
    ck = rep.get("checkpoint", {})
    cfg = rep["config"]
    st.caption(f'Report generated {rep.get("generated_at")} · seeds {rep["seeds"]} · {cfg["duration"]} s per run · demand {cfg["demand_vph"]} veh/h · '
               f'fixed plan {rep["fixed_greens"]} s · DQN checkpoint {Path(str(ck.get("checkpoint", "?"))).name} '
               f'({ck.get("episodes_trained")} training episodes)')
    for w in rep.get("warnings", []):
        ui.alert(w, "amber", "Compatibility warning")
    wait_label = next((i for i in imp.index if i.startswith("Average waiting")), None)
    if wait_label is not None and imp.loc[wait_label, "improvement_%"] == imp.loc[wait_label, "improvement_%"]:
        v = float(imp.loc[wait_label, "improvement_%"])
        ui.alert(f'Average waiting time: the DQN is {abs(v):.1f}% {"lower (better)" if v > 0 else "higher (worse)"} than fixed-time on average; '
                 f'DQN better in {imp.loc[wait_label, "dqn_better_in_seeds"]} seeds.', "green" if v > 0 else "amber", "Headline (measured)")
    st.plotly_chart(_improvement_bar(imp), use_container_width=True, config=charts.PLOT_CONFIG, key="cmp_imp")
    st.plotly_chart(_metric_bars(imp), use_container_width=True, config=charts.PLOT_CONFIG, key="cmp_bars")
    ts = data["timeseries"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(_ts_line(ts, "queue_total", "Queue length over time (mean over seeds)"), use_container_width=True,
                    config=charts.PLOT_CONFIG, key="cmp_l1")
    c2.plotly_chart(_ts_line(ts, "avg_waiting_now", "Mean current waiting time (s)"), use_container_width=True,
                    config=charts.PLOT_CONFIG, key="cmp_l2")
    c3, c4 = st.columns(2)
    c3.plotly_chart(_ts_line(ts, "stopped_vehicles", "Stopped vehicles"), use_container_width=True,
                    config=charts.PLOT_CONFIG, key="cmp_l3")
    c4.plotly_chart(_ts_line(ts, "cum_arrived", "Cumulative vehicles arrived"), use_container_width=True,
                    config=charts.PLOT_CONFIG, key="cmp_l4")
    t1, t2, t3, t4 = st.tabs(["Comparison table", "Per-seed improvement %", "Context", "Raw runs"])
    with t1:
        st.dataframe(imp.round(2), use_container_width=True)
    with t2:
        st.dataframe(data["per_seed"].round(1), use_container_width=True)
    with t3:
        st.dataframe(data["context"].round(1), use_container_width=True)
        st.caption("Average travel time counts only vehicles that arrived; compare 'vehicles_in_network_end' to see whether a controller leaves vehicles stuck.")
    with t4:
        st.dataframe(data["summaries"], use_container_width=True, hide_index=True)
    d = st.columns(4)
    d[0].download_button("⬇  Comparison (CSV)", csv_bytes(imp), file_name="comparison.csv", mime="text/csv", use_container_width=True)
    d[1].download_button("⬇  Per-seed (CSV)", csv_bytes(data["per_seed"]), file_name="per_seed_improvement.csv", mime="text/csv", use_container_width=True)
    d[2].download_button("⬇  Raw runs (CSV)", csv_bytes(data["summaries"], index=False), file_name="summaries.csv", mime="text/csv", use_container_width=True)
    d[3].download_button("⬇  Full report (Markdown)", data["markdown"].encode("utf-8"), file_name="comparison_report.md",
                         mime="text/markdown", use_container_width=True)
    ui.alert("Results apply only to the demand, duration and seeds shown. The fixed-time plan is a hand-chosen baseline, so test a stronger one "
             "(Page 7). There is no significance test; read 'DQN better in seeds' for consistency. The DQN was trained on simulated SUMO traffic.",
             "blue", "How to read this")


def render() -> None:
    ui.page_header("Performance Comparison", "Fixed-time vs DQN control under identical demand, duration and seeds (route files hash-verified).", "📈")
    _controls(get_settings())
    _results()