"""pytest configuration: project root on sys.path and a session-wide SUMO network fixture."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from tests.helpers import SUMO_OK  # noqa: E402


@pytest.fixture(scope="session")
def sumo_network() -> Path:
    """Build (or reuse) the SUMO network once per test session."""
    if not SUMO_OK:
        pytest.skip("SUMO/TraCI not available")
    from simulation.build_network import ensure_network
    return Path(ensure_network())