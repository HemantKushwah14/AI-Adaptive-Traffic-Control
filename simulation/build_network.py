"""Generate the 4-way intersection (nodes, edges, connections), run netconvert,
and write simulation.sumocfg.

Run:  python -m simulation.build_network [--force]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from config.constants import (APPROACHES, ARM_LENGTH_M, INCOMING_EDGE, LANE_INDEX, MOVEMENTS,
                              NUM_LANES, OUTGOING_EDGE, SPEED_LIMIT_MS, TLS_ID, TURN_TARGET)
from config.settings import (DEFAULT_ROUTE_FILE, NET_FILE, SUMO_CONFIG_DIR, SUMOCFG_FILE,
                             SimulationConfig, ensure_dirs, find_sumo_binary)
from utils.logging_utils import get_logger

log = get_logger(__name__)

NOD_FILE = SUMO_CONFIG_DIR / "intersection.nod.xml"
EDG_FILE = SUMO_CONFIG_DIR / "intersection.edg.xml"
CON_FILE = SUMO_CONFIG_DIR / "intersection.con.xml"

XML_HEAD = '<?xml version="1.0" encoding="UTF-8"?>'


def write_node_file(path: Path = NOD_FILE) -> None:
    L = ARM_LENGTH_M
    coords = {"N": (0.0, L), "S": (0.0, -L), "E": (L, 0.0), "W": (-L, 0.0)}
    lines = [XML_HEAD, "<nodes>",
             f'    <node id="{TLS_ID}" x="0.0" y="0.0" type="traffic_light"/>']
    for a in APPROACHES:
        x, y = coords[a]
        lines.append(f'    <node id="{a}" x="{x}" y="{y}" type="priority"/>')
    lines.append("</nodes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_edge_file(path: Path = EDG_FILE) -> None:
    lines = [XML_HEAD, "<edges>"]
    for a in APPROACHES:
        lines.append(f'    <edge id="{INCOMING_EDGE[a]}" from="{a}" to="{TLS_ID}" '
                     f'numLanes="{NUM_LANES}" speed="{SPEED_LIMIT_MS}" priority="2"/>')
        lines.append(f'    <edge id="{OUTGOING_EDGE[a]}" from="{TLS_ID}" to="{a}" '
                     f'numLanes="{NUM_LANES}" speed="{SPEED_LIMIT_MS}" priority="2"/>')
    lines.append("</edges>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_connection_file(path: Path = CON_FILE) -> None:
    """One connection per incoming lane: lane0=right, lane1=straight, lane2=left."""
    lines = [XML_HEAD, "<connections>"]
    for a in APPROACHES:
        for m in MOVEMENTS:
            idx = LANE_INDEX[m]
            to_edge = OUTGOING_EDGE[TURN_TARGET[a][m]]
            lines.append(f'    <connection from="{INCOMING_EDGE[a]}" to="{to_edge}" '
                         f'fromLane="{idx}" toLane="{idx}"/>')
    lines.append("</connections>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sumocfg(path: Path = SUMOCFG_FILE) -> None:
    text = f"""{XML_HEAD}
<configuration>
    <input>
        <net-file value="{NET_FILE.name}"/>
        <route-files value="{DEFAULT_ROUTE_FILE.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1800"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    path.write_text(text, encoding="utf-8")


def run_netconvert() -> None:
    cmd = [find_sumo_binary("netconvert"),
           "--node-files", str(NOD_FILE), "--edge-files", str(EDG_FILE),
           "--connection-files", str(CON_FILE), "--output-file", str(NET_FILE),
           "--no-turnarounds", "true", "--tls.default-type", "static", "--no-warnings", "true"]
    log.info("Running netconvert ...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"netconvert failed (code {res.returncode}):\n{res.stdout}\n{res.stderr}")


def verify_network() -> None:
    """Check the generated network has the expected traffic light with 12 signal links."""
    try:
        import sumolib  # type: ignore
    except ImportError:
        log.warning("sumolib not importable; skipping network verification.")
        return
    net = sumolib.net.readNet(str(NET_FILE), withPrograms=True)
    tls = net.getTLS(TLS_ID)
    n_links = len(tls.getConnections())
    expected = len(APPROACHES) * len(MOVEMENTS)
    if n_links != expected:
        raise RuntimeError(f"Traffic light '{TLS_ID}' has {n_links} links, expected {expected}.")
    log.info("Network verified: traffic light '%s' controls %d links.", TLS_ID, n_links)


def build_network(force: bool = False) -> Path:
    """Build network + default routes + sumocfg. Skips work if the net already exists."""
    ensure_dirs()
    if NET_FILE.exists() and not force:
        return NET_FILE
    write_node_file()
    write_edge_file()
    write_connection_file()
    run_netconvert()
    verify_network()
    write_sumocfg()
    # default routes so the .sumocfg can also be opened manually in sumo-gui
    from simulation.route_generator import generate_routes
    generate_routes(SimulationConfig(), DEFAULT_ROUTE_FILE)
    log.info("Network ready: %s", NET_FILE)
    return NET_FILE


def ensure_network() -> Path:
    return build_network(force=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the SUMO intersection network.")
    parser.add_argument("--force", action="store_true", help="rebuild even if files exist")
    args = parser.parse_args()
    path = build_network(force=args.force)
    print(f"OK: {path}")
    print(f"Optional visual check:  sumo-gui -c {SUMOCFG_FILE}")