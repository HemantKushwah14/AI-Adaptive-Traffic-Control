"""Metric definitions (hand-computed expectations), improvement direction, report I/O."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.constants import APPROACHES
from config.settings import SimulationConfig
from evaluation.compare import (CONTEXT, METRICS, ComparisonResult, compare_controllers, pct_improvement)
from evaluation.emergency_report import ambulance_comparison, load_latest_emergency, save_emergency_report
from evaluation.metrics import MetricsCollector
from evaluation.reports import build_markdown, csv_bytes, df_to_markdown, load_latest, save_report


# ------------------------------------------------------------------ metrics
def veh(speed, waiting, accum, timeloss):
    return {"speed": speed, "waiting": waiting, "accum_waiting": accum, "timeloss": timeloss}


def snap(queue_total):
    return {"queue_total": queue_total, "waiting_total": 50.0,
            "queue": {a: queue_total / 4 for a in APPROACHES}, "waiting": {a: 12.5 for a in APPROACHES}}


def test_metrics_summary_matches_hand_computation():
    obs = SimpleNamespace(
        vehicles={"a": veh(0.0, 5.0, 30.0, 25.0)}, newly_finished=[], newly_departed=[],
        finished=[{"waiting": 10.0, "timeloss": 5.0, "travel_time": 100.0},
                  {"waiting": 20.0, "timeloss": 15.0, "travel_time": 200.0}])
    m = MetricsCollector()
    m.on_step(1.0, snap(4.0), obs, {"active_phase": 0, "mode": "GREEN", "phase_switches": 0})
    m.on_step(2.0, snap(6.0), obs, {"active_phase": 0, "mode": "GREEN", "phase_switches": 2})
    s = m.summary(3600.0, obs)
    assert s["vehicles_entered"] == 3 and s["vehicles_arrived"] == 2 and s["vehicles_in_network_end"] == 1
    assert s["avg_waiting_time_s"] == pytest.approx(20.0)         # (10 + 20 + 30) / 3, includes vehicles still inside
    assert s["avg_travel_time_s"] == pytest.approx(150.0)         # arrived vehicles only
    assert s["total_delay_s"] == pytest.approx(45.0)
    assert s["avg_queue_length"] == pytest.approx(5.0) and s["max_queue_length"] == 6.0
    assert s["avg_stopped_vehicles"] == 1.0 and s["throughput_veh_per_hour"] == pytest.approx(2.0)
    assert s["phase_switches"] == 2
    assert m.interval(1)["queue_mean"] == 6.0 and m.interval(99) == {"queue_mean": 0.0, "waiting_mean": 0.0,
                                                                     "arrived": 0, "stopped_mean": 0.0}


def test_metrics_empty_summary_is_safe():
    s = MetricsCollector().summary(10.0, SimpleNamespace(finished=[], vehicles={}))
    assert s["avg_waiting_time_s"] == 0.0 and s["throughput_veh_per_hour"] == 0.0


# ------------------------------------------------------------------ improvement maths
def test_pct_improvement_direction():
    assert pct_improvement("avg_waiting_time_s", 100.0, 80.0) == pytest.approx(20.0)
    assert pct_improvement("avg_waiting_time_s", 100.0, 120.0) == pytest.approx(-20.0)
    assert pct_improvement("throughput_veh_per_hour", 100.0, 120.0) == pytest.approx(20.0)
    assert pct_improvement("throughput_veh_per_hour", 100.0, 80.0) == pytest.approx(-20.0)
    assert np.isnan(pct_improvement("avg_waiting_time_s", 0.0, 5.0))


def make_result() -> ComparisonResult:
    rows = []
    for seed, dqn_wait in ((1, 80.0), (2, 90.0)):
        for ctrl in ("fixed_time", "dqn"):
            r = {"controller": ctrl, "seed": seed}
            r.update({m: 100.0 for m in METRICS})
            r.update({c: 10.0 for c in CONTEXT})
            r["throughput_veh_per_hour"] = 110.0 if ctrl == "dqn" else 100.0
            if ctrl == "dqn":
                r["avg_waiting_time_s"] = dqn_wait
            rows.append(r)
    ts = pd.DataFrame({"controller": ["fixed_time", "dqn"] * 2, "seed": [1, 1, 2, 2],
                       "time": [1.0] * 4, "queue_total": [1, 2, 3, 4]})
    return ComparisonResult(config=SimulationConfig().to_dict(), seeds=[1, 2], fixed_greens=[35, 10, 35, 10],
                            checkpoint={"checkpoint": "x.pt", "episodes_trained": 3, "trained_at": None},
                            summaries=pd.DataFrame(rows), timeseries=ts)


def test_improvement_table_values():
    t = make_result().improvement_table()
    w = t.loc["Average waiting time (s)"]
    assert w["fixed_time"] == 100.0 and w["dqn"] == pytest.approx(85.0)
    assert w["improvement_%"] == pytest.approx(15.0) and w["dqn_better_in_seeds"] == "2/2"
    thr = t.loc["Throughput (veh/h)"]
    assert thr["improvement_%"] == pytest.approx(10.0)
    assert t.loc["Total delay (s)", "improvement_%"] == 0.0 and t.loc["Total delay (s)", "dqn_better_in_seeds"] == "0/2"


def test_compare_validates_arguments_before_running():
    cfg = SimulationConfig()
    with pytest.raises(ValueError):
        compare_controllers(cfg, None, seeds=[])
    with pytest.raises(ValueError):
        compare_controllers(cfg, None, seeds=[1, 1])
    with pytest.raises(ValueError):
        compare_controllers(cfg, None, seeds=[-3])
    with pytest.raises(ValueError):
        compare_controllers(cfg, None, seeds=[1], fixed_greens=(1, 2, 3))


# ------------------------------------------------------------------ reports
def test_markdown_helpers():
    df = pd.DataFrame({"a": [1.234, 2.0]}, index=pd.Index(["x", "y"], name="m"))
    lines = df_to_markdown(df, digits=1).splitlines()
    assert lines[:3] == ["| m | a |", "|---|---|", "| x | 1.2 |"]
    assert csv_bytes(df).startswith(b"m,a")


def test_report_content_and_roundtrip(tmp_path):
    result = make_result()
    md = build_markdown(result)
    assert "Limitations" in md and "Average waiting time (s)" in md and "hash-verified" in md
    paths = save_report(result, out_root=tmp_path)
    for key in ("summaries", "timeseries", "improvement", "per_seed", "context", "json", "markdown"):
        assert paths[key].exists(), key
    loaded = load_latest(tmp_path)
    assert loaded["improvement"].loc["Average waiting time (s)", "improvement_%"] == pytest.approx(15.0)
    assert loaded["report"]["seeds"] == [1, 2] and "Limitations" in loaded["markdown"]
    assert load_latest(tmp_path / "nothing_here") is None


def test_emergency_report_helpers(tmp_path):
    off = pd.DataFrame([{"vehicle_id": "amb_1", "approach": "E", "required_phase": "EW_LEFT",
                         "detection_to_clear_s": 30.0, "accum_waiting_s": 12.0}])
    on = pd.DataFrame([{"vehicle_id": "amb_1", "detection_to_clear_s": 20.0, "accum_waiting_s": 2.0,
                        "wait_for_green_s": 4.0, "status": "cleared"}])
    table = ambulance_comparison(off, on)
    assert table.loc[0, "time_saved_s"] == 10.0 and table.loc[0, "green_after_detection_s"] == 4.0
    save_emergency_report({"base": "fixed"}, {"ambulance_comparison": table, "events_priority": on}, tmp_path)
    loaded = load_latest_emergency(tmp_path)
    assert loaded["report"]["base"] == "fixed" and len(loaded["ambulance_comparison"]) == 1
    assert load_latest_emergency(tmp_path / "none") is None