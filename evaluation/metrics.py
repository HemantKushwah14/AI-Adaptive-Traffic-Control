"""Per-step recording and summary KPIs (all computed from real TraCI measurements).

Definitions
-----------
avg_waiting_time_s : mean accumulated waiting time (speed < 0.1 m/s) per vehicle, over ALL
                     vehicles that entered the network (finished + still inside), so a
                     controller cannot look good by leaving cars stuck.
avg_travel_time_s  : mean trip duration of vehicles that ARRIVED.
total_delay_s      : sum of SUMO "time loss" (time lost vs. free-flow) over all entered vehicles.
avg_queue_length   : mean over steps of halting vehicles on the 12 incoming lanes.
avg_stopped_vehicles: mean over steps of vehicles with speed < 0.1 m/s in the network.
throughput_veh_per_hour: arrived vehicles / simulated hours.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config.constants import APPROACHES


class MetricsCollector:
    def __init__(self) -> None:
        self.records: List[dict] = []

    def on_step(self, sim_time: float, snapshot: dict, observer, signal_info: dict) -> dict:
        vehicles = observer.vehicles
        n = len(vehicles)
        stopped = sum(1 for v in vehicles.values() if v["speed"] < 0.1)
        avg_wait_now = float(np.mean([v["waiting"] for v in vehicles.values()])) if n else 0.0
        rec = {
            "time": sim_time,
            "vehicles_in_network": n,
            "stopped_vehicles": stopped,
            "avg_waiting_now": avg_wait_now,
            "queue_total": snapshot["queue_total"],
            "lane_waiting_total": snapshot["waiting_total"],
            "arrived_step": len(observer.newly_finished),
            "departed_step": len(observer.newly_departed),
            "cum_arrived": len(observer.finished),
            "active_phase": signal_info["active_phase"],
            "mode": signal_info["mode"],
            "phase_switches": signal_info["phase_switches"],
        }
        for a in APPROACHES:
            rec[f"queue_{a}"] = snapshot["queue"][a]
            rec[f"waiting_{a}"] = snapshot["waiting"][a]
        self.records.append(rec)
        return rec

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def interval(self, start_index: int) -> Dict[str, float]:
        """Aggregate over records[start_index:] (used for RL rewards)."""
        sl = self.records[start_index:]
        if not sl:
            return {"queue_mean": 0.0, "waiting_mean": 0.0, "arrived": 0, "stopped_mean": 0.0}
        return {
            "queue_mean": float(np.mean([r["queue_total"] for r in sl])),
            "waiting_mean": float(np.mean([r["lane_waiting_total"] for r in sl])),
            "arrived": int(sum(r["arrived_step"] for r in sl)),
            "stopped_mean": float(np.mean([r["stopped_vehicles"] for r in sl])),
        }

    def summary(self, sim_time: float, observer) -> Dict[str, float]:
        finished = observer.finished
        inside = list(observer.vehicles.values())
        entered = len(finished) + len(inside)
        wait_all = [f["waiting"] for f in finished] + [v["accum_waiting"] for v in inside]
        loss_all = [f["timeloss"] for f in finished] + [v["timeloss"] for v in inside]
        df = self.to_dataframe()
        hours = max(sim_time, 1e-9) / 3600.0
        return {
            "sim_time_s": float(sim_time),
            "vehicles_entered": int(entered),
            "vehicles_arrived": int(len(finished)),
            "vehicles_in_network_end": int(len(inside)),
            "avg_waiting_time_s": float(np.mean(wait_all)) if wait_all else 0.0,
            "avg_travel_time_s": float(np.mean([f["travel_time"] for f in finished])) if finished else 0.0,
            "total_delay_s": float(np.sum(loss_all)) if loss_all else 0.0,
            "avg_queue_length": float(df["queue_total"].mean()) if len(df) else 0.0,
            "max_queue_length": float(df["queue_total"].max()) if len(df) else 0.0,
            "avg_stopped_vehicles": float(df["stopped_vehicles"].mean()) if len(df) else 0.0,
            "throughput_veh_per_hour": float(len(finished) / hours),
            "phase_switches": int(df["phase_switches"].iloc[-1]) if len(df) else 0,
        }