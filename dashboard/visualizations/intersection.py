"""Top-down intersection figure built ONLY from real data:
  * road/lane geometry  <- sumolib reading sumo_config/intersection.net.xml
  * vehicles            <- TraCI position / angle / type from the session snapshot
  * signal bars         <- the signal state string actually applied to SUMO
No independent animation exists, so the picture cannot disagree with the simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go

from config.constants import APPROACH_NAMES, APPROACHES, LANE_INDEX, TLS_ID
from config.settings import NET_FILE
from dashboard.styles.theme import COLORS

Pt = Tuple[float, float]
ROAD_FILL, ROAD_EDGE = "#16224a", "#2b3d77"
SIG_COLOR = {"G": COLORS["green"], "g": COLORS["green"], "y": COLORS["amber"], "Y": COLORS["amber"],
             "r": COLORS["red"], "R": COLORS["red"]}
IDLE_BAR = "#4a5578"
APPROACH_DIR = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}
VEH_DIMS = {"car": (4.5, 1.8), "motorcycle": (2.2, 0.8), "bus": (12.0, 2.5), "truck": (9.0, 2.4),
            "ambulance": (6.0, 2.0)}
VEH_COLORS = {"car": "#22b8ff", "motorcycle": "#ffc933", "bus": "#2fe39a", "truck": "#b48cff",
              "ambulance": "#ff3b4e"}
AMB_FLASH = "#3b82ff"
TURN_OFFSET = {"right": 40.0, "straight": 0.0, "left": -40.0}


@dataclass(frozen=True)
class LaneGeom:
    lane_id: str
    approach: str
    index: int
    incoming: bool
    p0: Pt
    p1: Pt
    width: float
    n_lanes: int


@dataclass(frozen=True)
class NetGeometry:
    lanes: Tuple[LaneGeom, ...]
    junction: Tuple[Pt, ...]
    center: Pt = (0.0, 0.0)      # junction centre in SUMO coordinates (already subtracted from lanes/junction)


@lru_cache(maxsize=2)
def _load(path: str, mtime: float) -> NetGeometry:
    try:
        import sumolib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sumolib is not installed: pip install sumolib eclipse-sumo") from exc
    net = sumolib.net.readNet(path)
    try:
        c = net.getNode(TLS_ID).getCoord()
        cx, cy = float(c[0]), float(c[1])
    except Exception:
        cx = cy = 0.0
    lanes: List[LaneGeom] = []
    for edge in net.getEdges():
        eid = edge.getID()
        if len(eid) != 3 or eid[1] != "2":
            continue
        incoming = eid.endswith("2C")
        approach = eid[0] if incoming else eid[-1]
        if approach not in APPROACHES:
            continue
        el = edge.getLanes()
        for lane in el:
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            lanes.append(LaneGeom(lane.getID(), approach, int(lane.getIndex()), incoming,
                                  (float(shape[0][0]) - cx, float(shape[0][1]) - cy),
                                  (float(shape[-1][0]) - cx, float(shape[-1][1]) - cy),
                                  float(lane.getWidth()), len(el)))
    try:
        junction = tuple((float(p[0]) - cx, float(p[1]) - cy) for p in net.getNode(TLS_ID).getShape())
    except Exception:
        junction = ()
    if len(junction) < 3:
        junction = ((-11.0, -11.0), (11.0, -11.0), (11.0, 11.0), (-11.0, 11.0))
    if not lanes:
        raise RuntimeError("No lanes found in the SUMO network. Rebuild: python -m simulation.build_network --force")
    return NetGeometry(tuple(lanes), junction, (cx, cy))


def load_geometry() -> NetGeometry:
    if not NET_FILE.exists():
        raise FileNotFoundError("SUMO network missing. Run: python -m simulation.build_network")
    return _load(str(NET_FILE), NET_FILE.stat().st_mtime)


# ------------------------------------------------------------------ drawing helpers
def _unit(a: Pt, b: Pt) -> Pt:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _lane_poly(l: LaneGeom) -> List[Pt]:
    d = _unit(l.p0, l.p1)
    n, h = (-d[1], d[0]), l.width / 2.0
    return [(l.p0[0] + n[0] * h, l.p0[1] + n[1] * h), (l.p1[0] + n[0] * h, l.p1[1] + n[1] * h),
            (l.p1[0] - n[0] * h, l.p1[1] - n[1] * h), (l.p0[0] - n[0] * h, l.p0[1] - n[1] * h)]


def _poly_trace(polys: List[List[Pt]], fill: str, line: str, width: float = 0.6) -> go.Scatter:
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    for poly in polys:
        pts = list(poly) + [poly[0]]
        xs += [p[0] for p in pts] + [None]
        ys += [p[1] for p in pts] + [None]
    return go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=fill,
                      line=dict(color=line, width=width), hoverinfo="skip", showlegend=False)


def _seg_trace(segs: List[Tuple[Pt, Pt]], color: str, width: float, dash: str = "solid",
               opacity: float = 1.0) -> go.Scatter:
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    for a, b in segs:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash),
                      opacity=opacity, hoverinfo="skip", showlegend=False)


def vehicle_polygon(x: float, y: float, angle_deg: float, length: float, width: float,
                    scale: float = 1.0) -> List[Pt]:
    """SUMO position = front-centre of the vehicle; angle = degrees clockwise from north."""
    a = math.radians(angle_deg)
    fx, fy = math.sin(a), math.cos(a)           # forward unit vector
    rx, ry = fy, -fx                            # right unit vector
    L, W = length * scale, width * scale
    bx, by = x - fx * L, y - fy * L
    return [(x - rx * W / 2, y - ry * W / 2), (x + rx * W / 2, y + ry * W / 2),
            (bx + rx * W / 2, by + ry * W / 2), (bx - rx * W / 2, by - ry * W / 2)]


def _lane_signal_chars(snap: Optional[dict]) -> Dict[str, str]:
    if not snap:
        return {}
    links, state = snap["signal"]["links"], snap["signal"]["state_string"]
    return {f"{a}2C_{i}": state[k] for k, (a, i) in enumerate(links) if k < len(state)}


def build_intersection_figure(snap: Optional[dict], view_range: float = 110.0,
                              vehicle_scale: float = 1.2, height: int = 650) -> go.Figure:
    geo = load_geometry()
    R = float(view_range)
    fig = go.Figure()

    # road surface + junction
    fig.add_trace(_poly_trace([_lane_poly(l) for l in geo.lanes] + [list(geo.junction)],
                              ROAD_FILL, ROAD_EDGE))
    # lane markings
    dash, center, edge = [], [], []
    for l in geo.lanes:
        d = _unit(l.p0, l.p1)
        n, h = (-d[1], d[0]), l.width / 2.0
        left = ((l.p0[0] + n[0] * h, l.p0[1] + n[1] * h), (l.p1[0] + n[0] * h, l.p1[1] + n[1] * h))
        (center if l.index == l.n_lanes - 1 else dash).append(left)
        if l.index == 0:
            edge.append(((l.p0[0] - n[0] * h, l.p0[1] - n[1] * h), (l.p1[0] - n[0] * h, l.p1[1] - n[1] * h)))
    fig.add_trace(_seg_trace(dash, "rgba(220,230,255,0.45)", 1, "dash"))
    fig.add_trace(_seg_trace(center, COLORS["amber"], 2))
    fig.add_trace(_seg_trace(edge, "rgba(190,205,255,0.55)", 2))

    # lane-movement arrows near the stop line
    ax, ay, ang = [], [], []
    for l in (g for g in geo.lanes if g.incoming):
        d = _unit(l.p0, l.p1)
        heading = math.degrees(math.atan2(d[0], d[1]))
        move = {v: k for k, v in LANE_INDEX.items()}.get(l.index, "straight")
        ax.append(l.p1[0] - d[0] * 16.0)
        ay.append(l.p1[1] - d[1] * 16.0)
        ang.append(heading + TURN_OFFSET[move])
    fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="arrow", size=11, angle=ang, color="rgba(220,230,255,0.55)")))

    # signal stop bars (state string applied to SUMO)
    chars = _lane_signal_chars(snap)
    groups: Dict[str, List[Tuple[Pt, Pt]]] = {}
    for l in (g for g in geo.lanes if g.incoming):
        d = _unit(l.p0, l.p1)
        n, h = (-d[1], d[0]), l.width / 2.0
        color = SIG_COLOR.get(chars.get(l.lane_id, ""), IDLE_BAR)
        groups.setdefault(color, []).append(((l.p1[0] + n[0] * h, l.p1[1] + n[1] * h),
                                             (l.p1[0] - n[0] * h, l.p1[1] - n[1] * h)))
    for color, segs in groups.items():
        fig.add_trace(_seg_trace(segs, color, 13, opacity=0.22))
        fig.add_trace(_seg_trace(segs, color, 5))

    # vehicles
    # vehicles (SUMO coordinates -> centred view coordinates)
    cx, cy = geo.center
    t = float(snap["time"]) if snap else 0.0
    flash = int(t * 2) % 2 == 1
    by_type: Dict[str, List[List[Pt]]] = {}
    amb_pts: List[Pt] = []
    for v in (snap["vehicles"] if snap else []):
        vx, vy = v["x"] - cx, v["y"] - cy
        if abs(vx) > R + 25 or abs(vy) > R + 25:
            continue
        L, W = VEH_DIMS.get(v["type"], VEH_DIMS["car"])
        by_type.setdefault(v["type"], []).append(
            vehicle_polygon(vx, vy, v["angle"], L, W, vehicle_scale))
        if v["type"] == "ambulance":
            amb_pts.append((vx, vy))
    for vt, polys in by_type.items():
        color = (AMB_FLASH if flash else VEH_COLORS["ambulance"]) if vt == "ambulance" \
            else VEH_COLORS.get(vt, VEH_COLORS["car"])
        fig.add_trace(_poly_trace(polys, color, "rgba(255,255,255,0.85)", 0.8))
    if amb_pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in amb_pts], y=[p[1] for p in amb_pts], mode="markers",
                                 hoverinfo="skip", showlegend=False,
                                 marker=dict(size=34, color="rgba(255,77,94,0.16)",
                                             line=dict(color=COLORS["red"], width=2))))
        
    # approach labels
    notes = []
    for a in APPROACHES:
        info = snap["approaches"].get(a) if snap else None
        if info is None:
            text, col = f"<b>{APPROACH_NAMES[a]}</b>", COLORS["muted"]
        else:
            text = f"<b>{APPROACH_NAMES[a]}</b>  {info['count']:.0f} veh · queue {info['queue']:.0f}"
            col = COLORS["green"] if info["green"] else "#ff8a96"
        dx, dy = APPROACH_DIR[a]
        notes.append(dict(x=dx * R * 0.86, y=dy * R * 0.86, text=text, showarrow=False,
                          font=dict(size=12, color=col), bgcolor="rgba(7,15,36,0.80)",
                          bordercolor=col, borderwidth=1, borderpad=4))
    if snap:
        sg = snap["signal"]
        hud = (f"<b>t = {t:.0f} s</b> &nbsp;·&nbsp; {sg['phase_names'][sg['active_phase']]}"
               f" &nbsp;·&nbsp; {sg['mode'].replace('_', '-')}")
    else:
        hud = "<b>IDLE</b> &nbsp;·&nbsp; press Start"
    notes.append(dict(xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
                      text=hud, showarrow=False, font=dict(size=13, color=COLORS["text"]),
                      bgcolor="rgba(7,15,36,0.80)", bordercolor=COLORS["border"], borderwidth=1, borderpad=6))
    legend = "  ".join(f'<span style="color:{VEH_COLORS[k]}">■</span> {k}' for k in VEH_COLORS)
    notes.append(dict(xref="paper", yref="paper", x=0.01, y=0.01, xanchor="left", yanchor="bottom",
                      text=legend, showarrow=False, font=dict(size=11, color=COLORS["muted"]),
                      bgcolor="rgba(7,15,36,0.80)", borderpad=5))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision="intersection",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["plot_bg"], annotations=notes,
        font=dict(family="Inter, Segoe UI, sans-serif", color=COLORS["text"]),
        xaxis=dict(range=[-R, R], visible=False, fixedrange=True),
        yaxis=dict(range=[-R, R], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True))
    return fig