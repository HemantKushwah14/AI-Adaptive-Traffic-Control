"""Seeded vehicle-route generation. Same SimulationConfig (incl. seed) => identical file,
so fixed-time and DQN controllers face exactly the same demand."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from config.constants import (APPROACHES, EMERGENCY_TYPE, INCOMING_EDGE, MOVEMENTS, OUTGOING_EDGE,
                              TURN_TARGET)
from config.settings import SimulationConfig, ensure_dirs

VTYPES_XML = """    <vType id="car" vClass="passenger" length="4.5" width="1.8" accel="2.6" decel="4.5" sigma="0.5" minGap="2.5" maxSpeed="13.89" color="0,170,255" guiShape="passenger"/>
    <vType id="motorcycle" vClass="passenger" length="2.2" width="0.8" accel="3.5" decel="5.0" sigma="0.5" minGap="1.5" maxSpeed="13.89" color="255,200,0" guiShape="motorcycle"/>
    <vType id="bus" vClass="passenger" length="12.0" width="2.5" accel="1.2" decel="4.0" sigma="0.4" minGap="3.0" maxSpeed="11.11" color="0,200,120" guiShape="bus"/>
    <vType id="truck" vClass="passenger" length="9.0" width="2.4" accel="1.3" decel="4.0" sigma="0.4" minGap="3.0" maxSpeed="11.11" color="170,120,255" guiShape="truck"/>
    <vType id="ambulance" vClass="emergency" length="6.0" width="2.0" accel="3.0" decel="5.0" sigma="0.2" minGap="2.0" maxSpeed="16.67" color="255,40,60" guiShape="emergency"/>
"""


@dataclass
class RouteSummary:
    path: Path
    total_vehicles: int
    per_approach: Dict[str, int]
    ambulances: List[dict] = field(default_factory=list)


def _choice(rng: random.Random, probs: Dict[str, float]) -> str:
    keys = sorted(probs)
    r, acc = rng.random() * sum(probs[k] for k in keys), 0.0
    for k in keys:
        acc += probs[k]
        if r <= acc:
            return k
    return keys[-1]


def generate_routes(cfg: SimulationConfig, output_path: Path) -> RouteSummary:
    """Write a SUMO route file and return a summary (including ambulance schedule)."""
    cfg.validate()
    ensure_dirs()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    rows: List[list] = []  # [depart, approach, movement, vtype]
    for a in APPROACHES:
        rate = cfg.demand_vph[a] / 3600.0
        if rate <= 0:
            continue
        t = 0.0
        while True:
            t += rng.expovariate(rate)
            if t >= cfg.duration:
                break
            rows.append([round(t, 2), a, _choice(rng, cfg.turn_probs), _choice(rng, cfg.vehicle_mix)])
    rows.sort(key=lambda r: (r[0], r[1]))
    if cfg.max_vehicles is not None:
        rows = rows[: cfg.max_vehicles]

    if cfg.emergency_probability > 0:
        em_rng = random.Random(cfg.seed + 10007)  # separate stream: base traffic unaffected
        for r in rows:
            if em_rng.random() < cfg.emergency_probability:
                r[3] = EMERGENCY_TYPE
    for t, a, m in cfg.emergency_events:
        rows.append([round(float(t), 2), a, m, EMERGENCY_TYPE])
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    # route definitions
    route_lines = []
    for a in APPROACHES:
        for m in MOVEMENTS:
            route_lines.append(f'    <route id="r_{a}_{m}" edges="{INCOMING_EDGE[a]} '
                               f'{OUTGOING_EDGE[TURN_TARGET[a][m]]}"/>')

    per_approach = {a: 0 for a in APPROACHES}
    ambulances: List[dict] = []
    veh_lines = []
    for i, (depart, a, m, vt) in enumerate(rows):
        is_amb = vt == EMERGENCY_TYPE
        vid = f"amb_{i}" if is_amb else f"veh_{i}"
        per_approach[a] += 1
        if is_amb:
            ambulances.append({"id": vid, "depart": depart, "approach": a, "movement": m})
        veh_lines.append(f'    <vehicle id="{vid}" type="{vt}" route="r_{a}_{m}" depart="{depart}" '
                         f'departLane="best" departSpeed="max"/>')

    text = ('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n' + VTYPES_XML
            + "\n".join(route_lines) + "\n" + "\n".join(veh_lines) + "\n</routes>\n")
    output_path.write_text(text, encoding="utf-8")
    return RouteSummary(output_path, len(rows), per_approach, ambulances)