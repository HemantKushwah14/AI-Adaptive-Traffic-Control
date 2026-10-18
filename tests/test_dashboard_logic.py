"""Dashboard logic that needs neither a browser nor SUMO: background jobs, settings, geometry."""
from __future__ import annotations

import time

import pytest

from dashboard.jobs import BackgroundJob, JobState, format_events, parse_events
from dashboard.sim_settings import build_sim_config, default_settings, make_controller
from dashboard.visualizations.intersection import vehicle_polygon
from reinforcement_learning.dqn_agent import CheckpointError
from simulation.fixed_time_controller import FixedTimeController


# ------------------------------------------------------------------ jobs
def wait_state(job, states, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        snap = job.snapshot()
        if snap["state"] in states:
            return snap
        time.sleep(0.02)
    pytest.fail(f"job stayed in {job.snapshot()['state']}, expected {states}")


def test_job_success_error_and_cancel():
    def quick(job, x):
        job.report(1, 1, "ok")
        return x * 2

    def boom(job):
        raise RuntimeError("boom")

    def slow(job):
        for i in range(1000):
            job.report(i, 1000, "working")
            time.sleep(0.01)

    ok = BackgroundJob("ok")
    assert ok.start(quick, 21)
    snap = wait_state(ok, {JobState.DONE.value})
    assert snap["result"] == 42 and snap["error"] is None

    bad = BackgroundJob("bad")
    bad.start(boom)
    assert wait_state(bad, {JobState.ERROR.value})["error"] == "boom"

    job = BackgroundJob("slow")
    assert job.start(slow) is True
    assert job.start(slow) is False, "a running job must not be started twice"
    time.sleep(0.05)
    job.cancel()
    wait_state(job, {JobState.CANCELLED.value})


# ------------------------------------------------------------------ event parsing
def test_event_parsing_roundtrip_and_errors():
    assert parse_events("100:N:straight, 260:E:left") == [(100.0, "N", "straight"), (260.0, "E", "left")]
    assert parse_events("100:n:Straight") == [(100.0, "N", "straight")] and parse_events("") == []
    assert format_events([(100.0, "N", "straight")]) == "100:N:straight"
    for bad in ("100:N", "abc:N:left"):
        with pytest.raises(ValueError):
            parse_events(bad)


# ------------------------------------------------------------------ settings and controller factory
def test_default_settings_build_a_valid_config():
    s = default_settings()
    cfg = build_sim_config(s)
    assert cfg.duration == s["duration"] and cfg.demand_vph == s["demand_vph"]


def test_invalid_settings_are_rejected():
    s = default_settings()
    s.update(min_green=20, max_green=10)
    with pytest.raises(ValueError):
        build_sim_config(s)


def test_make_controller_variants(tmp_path):
    s = default_settings()
    s.update(controller="fixed_time", emergency_priority=False)
    base, ctrl, warns = make_controller(s, build_sim_config(s))
    assert isinstance(base, FixedTimeController) and ctrl.enabled is False and warns == []
    s.update(controller="dqn", checkpoint=str(tmp_path / "nope.pt"))
    with pytest.raises(CheckpointError):
        make_controller(s, build_sim_config(s))
    s.update(controller="mystery")
    with pytest.raises(ValueError):
        make_controller(s, build_sim_config(s))


# ------------------------------------------------------------------ vehicle drawing geometry
def area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % 4][1] - poly[(i + 1) % 4][0] * poly[i][1] for i in range(4))) / 2


@pytest.mark.parametrize("angle", [0, 37, 90, 180, 271])
def test_vehicle_polygon_size_and_anchor(angle):
    poly = vehicle_polygon(5.0, -3.0, angle, 4.5, 1.8, 1.0)
    assert area(poly) == pytest.approx(8.1)
    front = ((poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2)
    rear = ((poly[2][0] + poly[3][0]) / 2, (poly[2][1] + poly[3][1]) / 2)
    assert front == pytest.approx((5.0, -3.0)), "SUMO position is the FRONT-centre of the vehicle"
    assert ((front[0] - rear[0]) ** 2 + (front[1] - rear[1]) ** 2) ** 0.5 == pytest.approx(4.5)
    assert area(vehicle_polygon(0, 0, angle, 4.5, 1.8, 2.0)) == pytest.approx(32.4)


def test_vehicle_heading_north_extends_south_of_front():
    poly = vehicle_polygon(0.0, 0.0, 0.0, 4.5, 1.8, 1.0)
    assert max(p[1] for p in poly) == pytest.approx(0.0) and min(p[1] for p in poly) == pytest.approx(-4.5)