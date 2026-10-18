"""SimulationConfig validation and seeded route generation (no SUMO needed)."""
from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET

import pytest

from config.constants import APPROACHES, MOVEMENTS
from config.settings import SimulationConfig
from simulation.route_generator import generate_routes

BAD = {
    "short_duration": {"duration": 10},
    "bad_step": {"step_length": 2.0},
    "negative_seed": {"seed": -1},
    "missing_demand": {"demand_vph": {"N": 500.0, "S": 500.0, "E": 500.0}},
    "excess_demand": {"demand_vph": {a: 5000.0 for a in APPROACHES}},
    "turns_not_one": {"turn_probs": {"right": 0.5, "straight": 0.5, "left": 0.5}},
    "unknown_vehicle": {"vehicle_mix": {"car": 0.5, "tank": 0.5}},
    "bad_cap": {"max_vehicles": 0},
    "emergency_too_high": {"emergency_probability": 0.5},
    "event_after_end": {"emergency_events": [(99999.0, "N", "straight")]},
    "event_bad_approach": {"emergency_events": [(10.0, "X", "straight")]},
    "min_green_small": {"min_green": 2},
    "max_not_above_min": {"max_green": 10},
    "yellow_zero": {"yellow_time": 0},
    "negative_all_red": {"all_red_time": -1},
    "interval_above_min": {"decision_interval": 11},
}


def test_default_config_is_valid():
    SimulationConfig().validate()


@pytest.mark.parametrize("changes", list(BAD.values()), ids=list(BAD))
def test_invalid_settings_are_rejected(changes):
    cfg = dataclasses.replace(SimulationConfig(), **changes)
    with pytest.raises(ValueError, match="Invalid simulation settings"):
        cfg.validate()


def test_all_errors_are_reported_together():
    cfg = dataclasses.replace(SimulationConfig(), duration=10, min_green=2)
    with pytest.raises(ValueError) as exc:
        cfg.validate()
    assert "duration" in str(exc.value) and "min_green" in str(exc.value)


def test_uniform_demand_constructor():
    cfg = SimulationConfig.with_uniform_demand(650, duration=300)
    assert all(cfg.demand_vph[a] == 650.0 for a in APPROACHES)
    cfg.validate()


# ------------------------------------------------------------------ routes
def _root(path):
    return ET.parse(path).getroot()


def test_routes_are_deterministic(tmp_path):
    cfg = SimulationConfig.with_uniform_demand(400, duration=300, seed=7)
    a = generate_routes(cfg, tmp_path / "a.rou.xml")
    b = generate_routes(cfg, tmp_path / "b.rou.xml")
    assert a.path.read_bytes() == b.path.read_bytes()
    assert a.total_vehicles == b.total_vehicles > 0


def test_different_seed_gives_different_routes(tmp_path):
    a = generate_routes(SimulationConfig.with_uniform_demand(400, duration=300, seed=1), tmp_path / "a.rou.xml")
    b = generate_routes(SimulationConfig.with_uniform_demand(400, duration=300, seed=2), tmp_path / "b.rou.xml")
    assert a.path.read_bytes() != b.path.read_bytes()


def test_higher_demand_generates_more_vehicles(tmp_path):
    low = generate_routes(SimulationConfig.with_uniform_demand(200, duration=600, seed=1), tmp_path / "l.rou.xml")
    high = generate_routes(SimulationConfig.with_uniform_demand(800, duration=600, seed=1), tmp_path / "h.rou.xml")
    assert high.total_vehicles > low.total_vehicles


def test_zero_demand_approach_has_no_vehicles(tmp_path):
    cfg = SimulationConfig.with_uniform_demand(400, duration=300, seed=3)
    cfg.demand_vph["N"] = 0.0
    s = generate_routes(cfg, tmp_path / "z.rou.xml")
    assert s.per_approach["N"] == 0 and sum(s.per_approach.values()) == s.total_vehicles


def test_max_vehicles_caps_total(tmp_path):
    cfg = SimulationConfig.with_uniform_demand(800, duration=600, seed=1, max_vehicles=25)
    assert generate_routes(cfg, tmp_path / "c.rou.xml").total_vehicles == 25


def test_route_file_structure(tmp_path):
    cfg = SimulationConfig.with_uniform_demand(400, duration=300, seed=4)
    s = generate_routes(cfg, tmp_path / "x.rou.xml")
    root = _root(s.path)
    route_ids = {r.get("id") for r in root.findall("route")}
    assert route_ids == {f"r_{a}_{m}" for a in APPROACHES for m in MOVEMENTS}
    vehicles = root.findall("vehicle")
    assert len(vehicles) == s.total_vehicles
    assert all(v.get("route") in route_ids for v in vehicles)
    departs = [float(v.get("depart")) for v in vehicles]
    assert departs == sorted(departs), "SUMO requires departures in ascending order"
    assert {v.get("type") for v in vehicles} <= {"car", "motorcycle", "bus", "truck"}
    assert {t.get("id") for t in root.findall("vType")} >= {"car", "motorcycle", "bus", "truck", "ambulance"}


def test_scheduled_ambulance(tmp_path):
    cfg = SimulationConfig.with_uniform_demand(300, duration=300, seed=5, emergency_events=[(100.0, "N", "straight")])
    s = generate_routes(cfg, tmp_path / "e.rou.xml")
    assert len(s.ambulances) == 1
    amb = s.ambulances[0]
    assert (amb["approach"], amb["movement"], amb["depart"]) == ("N", "straight", 100.0)
    xml_amb = [v for v in _root(s.path).findall("vehicle") if v.get("type") == "ambulance"]
    assert len(xml_amb) == 1 and xml_amb[0].get("id") == amb["id"] and xml_amb[0].get("route") == "r_N_straight"


def test_emergency_probability_keeps_base_timing(tmp_path):
    base = generate_routes(SimulationConfig.with_uniform_demand(500, duration=900, seed=9), tmp_path / "b.rou.xml")
    amb = generate_routes(SimulationConfig.with_uniform_demand(500, duration=900, seed=9, emergency_probability=0.2),
                          tmp_path / "a.rou.xml")
    dep = lambda p: [v.get("depart") for v in _root(p).findall("vehicle")]  # noqa: E731
    assert dep(base.path) == dep(amb.path), "ambulance replacement must not change departure times"
    assert len(amb.ambulances) > 0 and len(base.ambulances) == 0


def test_invalid_config_is_rejected_by_generator(tmp_path):
    with pytest.raises(ValueError):
        generate_routes(dataclasses.replace(SimulationConfig(), duration=5), tmp_path / "bad.rou.xml")