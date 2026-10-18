"""Shared test helpers: SUMO availability marker and a FAKE TraCI connection used only to unit-test
the signal state machine (it never produces results shown to users)."""
from __future__ import annotations

import random
from types import SimpleNamespace
from typing import List, Optional

import pytest

from config.constants import APPROACHES, NUM_LANES


def _sumo_available() -> bool:
    try:
        from config.settings import ensure_traci_importable, find_sumo_binary
        find_sumo_binary("sumo")
        find_sumo_binary("netconvert")
        ensure_traci_importable()
        return True
    except Exception:
        return False


SUMO_OK = _sumo_available()
requires_sumo = pytest.mark.skipif(not SUMO_OK, reason="SUMO/TraCI not available (run scripts/check_environment.py)")


class FakeTrafficLight:
    def __init__(self, links: List[list]) -> None:
        self._links = links
        self.states: List[str] = []

    def getControlledLinks(self, tls_id: str) -> List[list]:
        return self._links

    def setRedYellowGreenState(self, tls_id: str, state: str) -> None:
        self.states.append(state)


def make_fake_conn(shuffle_seed: Optional[int] = None, drop_last: bool = False, empty_group: bool = False):
    """12 signal links (N,S,E,W x lanes 0..2) in the same format as TraCI getControlledLinks."""
    pairs = [(a, i) for a in APPROACHES for i in range(NUM_LANES)]
    links = [[(f"{a}2C_{i}", f"C2{a}_{i}", f":C_{k}")] for k, (a, i) in enumerate(pairs)]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(links)
    if drop_last:
        links = links[:-1]
    if empty_group:
        links[0] = []
    return SimpleNamespace(trafficlight=FakeTrafficLight(links))