"""Dark-themed Plotly charts built from the session time series (real TraCI data only)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config.constants import APPROACH_NAMES, APPROACHES, PHASE_NAMES
from dashboard.styles.theme import COLORS

APPROACH_COLOR = {"N": COLORS["cyan"], "S": COLORS["blue"], "E": COLORS["green"], "W": COLORS["purple"]}
PHASE_COLOR = [COLORS["cyan"], COLORS["purple"], COLORS["green"], COLORS["amber"]]
PLOT_CONFIG = {"displayModeBar": False}
STATIC_CONFIG = {"displayModeBar": False, "staticPlot": True}


def style(fig: go.Figure, height: int = 280, title: str = "", legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=14, color=COLORS["text"]), x=0.02, y=0.96),
        margin=dict(l=44, r=14, t=40 if title else 14, b=36), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, sans-serif", color=COLORS["muted"], size=12),
        showlegend=legend, uirevision="keep",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, font=dict(size=11)),
        hovermode="x unified")
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, title_text="simulation time (s)")
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


def empty_fig(message: str = "No data yet - start a simulation", height: int = 280, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=COLORS["muted"], size=14),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height, title, legend=False)


def _thin(df: pd.DataFrame, max_points: int = 1200) -> pd.DataFrame:
    return df if len(df) <= max_points else df.iloc[:: int(np.ceil(len(df) / max_points))]


def traffic_chart(df: pd.DataFrame, height: int = 280) -> go.Figure:
    if df.empty:
        return empty_fig(height=height, title="Traffic in the network")
    d = _thin(df)
    fig = go.Figure()
    for col, name, color in (("vehicles", "Vehicles in network", COLORS["cyan"]),
                             ("queue", "Queue (halting)", COLORS["amber"]),
                             ("stopped", "Stopped", COLORS["red"])):
        fig.add_trace(go.Scatter(x=d["time"], y=d[col], name=name, mode="lines", line=dict(color=color, width=2)))
    return style(fig, height, "Traffic in the network")


def waiting_chart(df: pd.DataFrame, height: int = 280) -> go.Figure:
    if df.empty:
        return empty_fig(height=height, title="Mean current waiting time")
    d = _thin(df)
    fig = go.Figure(go.Scatter(x=d["time"], y=d["avg_wait_now"], name="Mean waiting (s)", mode="lines",
                               fill="tozeroy", line=dict(color=COLORS["blue"], width=2),
                               fillcolor="rgba(47,123,255,0.18)"))
    return style(fig, height, "Mean current waiting time of vehicles in network (s)", legend=False)


def approach_queue_chart(df: pd.DataFrame, height: int = 280) -> go.Figure:
    if df.empty:
        return empty_fig(height=height, title="Queue length per approach")
    d = _thin(df)
    fig = go.Figure()
    for a in APPROACHES:
        fig.add_trace(go.Scatter(x=d["time"], y=d[f"queue_{a}"], name=APPROACH_NAMES[a], mode="lines",
                                 line=dict(color=APPROACH_COLOR[a], width=2)))
    return style(fig, height, "Queue length per approach (halting vehicles)")


def throughput_chart(df: pd.DataFrame, height: int = 280) -> go.Figure:
    """Arrivals in the trailing 60 s, expressed as veh/h (NaN for the first 60 s)."""
    if df.empty or "cum_arrived" not in df:
        return empty_fig(height=height, title="Throughput (trailing 60 s)")
    t, cum = df["time"].to_numpy(), df["cum_arrived"].to_numpy()
    idx = np.searchsorted(t, t - 60.0, side="left")
    thr = (cum - cum[idx]) * 60.0
    thr[t < 60.0] = np.nan
    d = pd.DataFrame({"time": t, "thr": thr})
    d = _thin(d)
    fig = go.Figure(go.Scatter(x=d["time"], y=d["thr"], mode="lines", name="veh/h",
                               line=dict(color=COLORS["green"], width=2), fill="tozeroy",
                               fillcolor="rgba(34,229,138,0.14)"))
    return style(fig, height, "Throughput, trailing 60 s (veh/h)", legend=False)


def phase_timeline(df: pd.DataFrame, height: int = 220) -> go.Figure:
    if df.empty:
        return empty_fig(height=height, title="Signal phase timeline")
    d = _thin(df, 1500)
    fig = go.Figure(go.Scatter(x=d["time"], y=d["phase"], mode="lines+markers", line=dict(shape="hv", color=COLORS["cyan"], width=2),
                               marker=dict(size=5, color=[PHASE_COLOR[int(p)] for p in d["phase"]]), name="phase"))
    fig.update_yaxes(tickvals=list(range(len(PHASE_NAMES))), ticktext=list(PHASE_NAMES), range=[-0.5, len(PHASE_NAMES) - 0.5])
    return style(fig, height, "Active signal phase (yellow / all-red gaps not shown)", legend=False)


def approach_bars(snap: Optional[dict], height: int = 260) -> go.Figure:
    if not snap or not snap.get("approaches"):
        return empty_fig(height=height, title="Current queue and vehicle count")
    names = [APPROACH_NAMES[a] for a in APPROACHES]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[snap["approaches"][a]["count"] for a in APPROACHES], name="Vehicles",
                         marker_color=COLORS["blue"]))
    fig.add_trace(go.Bar(x=names, y=[snap["approaches"][a]["queue"] for a in APPROACHES], name="Queue",
                         marker_color=COLORS["amber"]))
    fig.update_layout(barmode="group")
    fig = style(fig, height, "Current vehicles and queue per approach")
    fig.update_xaxes(title_text="")
    return fig