"""Signal state machine safety, tested against a FAKE TraCI connection (no SUMO)."""
from __future__ import annotations

import pytest

from config.constants import NUM_PHASES, PHASE_GREEN
from simulation.traci_controller import SignalManager, SignalMode, SimulationError
from tests.helpers import make_fake_conn


def make(order_seed=None, yellow=3.0, all_red=1.0):
    conn = make_fake_conn(order_seed)
    sm = SignalManager(conn, yellow_time=yellow, all_red_time=all_red)
    sm.initialise(0)
    return sm, conn


def green_approaches(sm, state):
    return {sm._signals[i][0] for i, ch in enumerate(state) if ch in "Gg"}


def axis_safe(approaches):
    return approaches <= {"N", "S"} or approaches <= {"E", "W"}


@pytest.mark.parametrize("seed", [None, 1, 2, 3])
def test_green_states_match_phase_definition_and_are_conflict_free(seed):
    sm, _ = make(seed)
    for phase in range(NUM_PHASES):
        state = sm.green_state(phase)
        assert len(state) == 12 and set(state) <= {"G", "r"}
        for i, (approach, lane) in enumerate(sm._signals):
            assert (state[i] == "G") == (lane in PHASE_GREEN[phase].get(approach, ())), (phase, approach, lane)
        assert green_approaches(sm, state) and axis_safe(green_approaches(sm, state))


def test_phase_change_goes_green_yellow_allred_green():
    sm, conn = make()
    conn.trafficlight.states.clear()
    assert sm.request_phase(2) is True
    states = [conn.trafficlight.states[-1]]
    for _ in range(4):
        sm.tick(1.0)
        states.append(conn.trafficlight.states[-1])
    assert states[0] == states[1] == states[2] == sm.yellow_state(0)
    assert states[3] == sm.all_red_state()
    assert states[4] == sm.green_state(2)
    assert all("G" not in s for s in states[:4]), "no green may be shown during clearance"
    assert all(axis_safe(green_approaches(sm, s)) for s in states)
    assert (sm.active_phase, sm.mode, sm.elapsed_green, sm.phase_switches) == (2, SignalMode.GREEN, 0.0, 1)


def test_yellow_only_for_old_greens():
    sm, _ = make()
    y = sm.yellow_state(0)
    assert set(y) <= {"y", "r"}
    for i, (approach, lane) in enumerate(sm._signals):
        assert (y[i] == "y") == (lane in PHASE_GREEN[0].get(approach, ()))


def test_same_phase_request_is_a_noop():
    sm, conn = make()
    n = len(conn.trafficlight.states)
    assert sm.request_phase(0) is False
    assert len(conn.trafficlight.states) == n and sm.phase_switches == 0


@pytest.mark.parametrize("bad", [-1, 4, 7])
def test_invalid_phase_is_rejected(bad):
    sm, _ = make()
    with pytest.raises(ValueError):
        sm.request_phase(bad)


def test_retarget_during_transition_is_safe():
    sm, conn = make()
    sm.request_phase(2)
    sm.tick(1.0)
    assert sm.request_phase(3) is True and sm.target_phase == 3
    for _ in range(10):
        sm.tick(1.0)
    assert sm.mode == SignalMode.GREEN and sm.active_phase == 3 and sm.phase_switches == 1
    assert all(axis_safe(green_approaches(sm, s)) for s in conn.trafficlight.states)


def test_zero_all_red_goes_straight_to_green():
    sm, conn = make(all_red=0.0)
    sm.request_phase(2)
    for _ in range(3):
        sm.tick(1.0)
    assert sm.mode == SignalMode.GREEN and sm.active_phase == 2
    assert sm.all_red_state() not in conn.trafficlight.states


def test_elapsed_green_accumulates_and_info():
    sm, _ = make()
    for _ in range(5):
        sm.tick(1.0)
    info = sm.info()
    assert info["elapsed_green"] == 5.0 and info["mode"] == "GREEN"
    assert info["green_approaches"] == ["N", "S"] and info["transition_remaining"] == 0.0


def test_mismatched_signal_links_are_detected():
    with pytest.raises(SimulationError):
        SignalManager(make_fake_conn(drop_last=True))
    with pytest.raises(SimulationError, match="controls no link"):
        SignalManager(make_fake_conn(empty_group=True))