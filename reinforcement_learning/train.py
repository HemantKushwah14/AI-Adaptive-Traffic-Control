"""DQN training on the real SUMO intersection (TraCI).

    python -m reinforcement_learning.train --episodes 2 --duration 300        # quick timing run
    python -m reinforcement_learning.train --episodes 80 --duration 900
    python -m reinforcement_learning.train --episodes 120 --resume

Per episode: seeded random per-approach demand (so the agent sees varied, asymmetric traffic).
Validation (greedy, fixed seeds, uniform demand) picks best.pt; the test seeds are untouched.
Ctrl+C is safe: last.pt is written before exit.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.constants import ACTION_DIM, APPROACHES
from config.rl_config import (BEST_CKPT, DQN_DIR, DQN_OUT_DIR, LAST_CKPT, MAX_TRAIN_EPISODES,
                              TRAIN_LOG, TRAIN_SEED_BASE, TRAIN_SUMMARY, VAL_SEED_BASE)
from config.settings import SimulationConfig, SumoNotFoundError, ensure_dirs
from reinforcement_learning.action_mask import legal_action_mask
from reinforcement_learning.dqn_agent import CheckpointError, DQNAgent, DQNConfig
from reinforcement_learning.evaluate import run_dqn_episode, run_fixed_episode
from reinforcement_learning.reward import RewardWeights, compute_reward
from simulation.sumo_environment import SumoEnvironment
from simulation.traci_controller import SimulationError
from utils.logging_utils import get_logger

log = get_logger(__name__)

LOG_FIELDS = ["episode", "seed", "demand_mean", "epsilon", "decisions", "total_reward", "mean_reward",
              "mean_loss", "mean_q", "avg_wait_s", "avg_queue", "throughput_vph", "phase_switches",
              "buffer", "wall_s", "val_dqn_wait_s", "val_fixed_wait_s"]


def episode_config(base: SimulationConfig, episode: int, demand_range) -> SimulationConfig:
    rng = np.random.default_rng(TRAIN_SEED_BASE + episode)
    lo, hi = demand_range
    demand = {a: float(rng.uniform(lo, hi)) for a in APPROACHES}
    return dataclasses.replace(base, seed=TRAIN_SEED_BASE + episode, demand_vph=demand)


def train_episode(env: SumoEnvironment, agent: DQNAgent, cfg: SimulationConfig,
                  weights: RewardWeights) -> Dict:
    env.cfg = cfg
    state = env.reset()
    total_r, losses, qs, decisions = 0.0, [], [], 0
    limit = int(cfg.duration / cfg.decision_interval) + 200
    all_legal = np.ones(ACTION_DIM, dtype=bool)
    while not env.done:
        if decisions >= limit:
            raise SimulationError("Training episode exceeded its step limit.")
        action = agent.select_action(state, legal_action_mask(env))
        next_state, done, info = env.step(action)
        reward, _ = compute_reward(info, weights, cfg.decision_interval)
        next_mask = all_legal if done else legal_action_mask(env)
        # `done` here is the simulation time limit, not a true terminal state -> keep bootstrapping.
        agent.remember(state, info["action_applied"], reward, next_state, next_mask, terminal=False)
        out = agent.learn()
        if out:
            losses.append(out["loss"])
            qs.append(out["mean_q"])
        total_r += reward
        decisions += 1
        state = next_state
    agent.episodes += 1
    summ = env.summary()
    return {"decisions": decisions, "total_reward": total_r,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "mean_q": float(np.mean(qs)) if qs else float("nan"), "summary": summ}


def validate(env: SumoEnvironment, agent: DQNAgent, seeds: List[int]) -> float:
    waits = [run_dqn_episode(env, agent, seed=s)[0]["avg_waiting_time_s"] for s in seeds]
    return float(np.mean(waits))


def main() -> int:
    p = argparse.ArgumentParser(description="Train the DQN traffic-signal agent on SUMO.")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--duration", type=int, default=900, help="simulated seconds per episode")
    p.add_argument("--demand-min", type=float, default=300.0)
    p.add_argument("--demand-max", type=float, default=800.0)
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--max-green", type=int, default=60)
    p.add_argument("--decision-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=str, default="128,128,64")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learn-start", type=int, default=500)
    p.add_argument("--target-update", type=int, default=500)
    p.add_argument("--eps-decay-steps", type=int, default=8_000)
    p.add_argument("--double-dqn", action="store_true")
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--val-episodes", type=int, default=2)
    p.add_argument("--val-demand", type=float, default=500.0)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    if not (1 <= a.episodes <= MAX_TRAIN_EPISODES):
        print(f"ERROR: --episodes must be within 1..{MAX_TRAIN_EPISODES}")
        return 1
    if not (0 < a.demand_min <= a.demand_max <= 3000):
        print("ERROR: need 0 < demand-min <= demand-max <= 3000")
        return 1
    try:
        hidden = tuple(int(x) for x in a.hidden.split(","))
        base = SimulationConfig(duration=a.duration, min_green=a.min_green, max_green=a.max_green,
                                decision_interval=a.decision_interval, seed=TRAIN_SEED_BASE)
        base.validate()
        val_cfg = SimulationConfig.with_uniform_demand(
            a.val_demand, duration=a.duration, min_green=a.min_green, max_green=a.max_green,
            decision_interval=a.decision_interval)
        val_cfg.validate()
        weights = RewardWeights()
        ensure_dirs()
        DQN_DIR.mkdir(parents=True, exist_ok=True)
        DQN_OUT_DIR.mkdir(parents=True, exist_ok=True)

        if a.resume and LAST_CKPT.exists():
            agent = DQNAgent.from_checkpoint(LAST_CKPT, device=a.device, with_buffer=True,
                                             restore_optimizer=True)
            print(f"Resumed from {LAST_CKPT} at episode {agent.episodes} "
                  "(replay buffer starts empty; network hyper-parameters come from the checkpoint).")
        else:
            if a.resume:
                print("No last.pt found - starting fresh.")
            agent = DQNAgent(DQNConfig(
                hidden=hidden, gamma=a.gamma, lr=a.lr, batch_size=a.batch_size,
                buffer_size=a.buffer_size, learn_start=a.learn_start, target_update_steps=a.target_update,
                eps_decay_steps=a.eps_decay_steps, double_dqn=a.double_dqn, seed=a.seed, device=a.device))
    except (ValueError, CheckpointError) as exc:
        print(f"ERROR: {exc}")
        return 1

    start_ep = agent.episodes
    if start_ep >= a.episodes:
        print(f"Checkpoint already has {start_ep} episodes >= --episodes {a.episodes}. Nothing to do.")
        return 0
    val_seeds = [VAL_SEED_BASE + i for i in range(max(1, a.val_episodes))]
    best_val = float(agent.meta.get("best_val_wait_s", float("inf")))
    new_log = not (a.resume and TRAIN_LOG.exists())
    t_start = time.time()
    last_val_dqn = last_val_fixed = float("nan")

    def meta_now(ep: int) -> dict:
        return {"episode": ep, "trained_at": datetime.now().isoformat(timespec="seconds"),
                "best_val_wait_s": best_val, "reward_weights": weights.to_dict(),
                "train_sim_config": base.to_dict(), "demand_range": [a.demand_min, a.demand_max],
                "val_demand": a.val_demand, "val_seeds": val_seeds}

    env = SumoEnvironment(base, label="train")
    val_env = SumoEnvironment(val_cfg, label="val")
    try:
        with open(TRAIN_LOG, "w" if new_log else "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if new_log:
                writer.writeheader()
            if start_ep == 0:
                fixed_waits = [run_fixed_episode(val_env, seed=s)["avg_waiting_time_s"] for s in val_seeds]
                last_val_fixed = float(np.mean(fixed_waits))
                print(f"Validation baseline (fixed-time, seeds {val_seeds}): avg wait {last_val_fixed:.1f}s")
            for ep in range(start_ep, a.episodes):
                t0 = time.time()
                cfg_ep = episode_config(base, ep, (a.demand_min, a.demand_max))
                res = train_episode(env, agent, cfg_ep, weights)
                s = res["summary"]
                row = {"episode": ep + 1, "seed": cfg_ep.seed,
                       "demand_mean": float(np.mean(list(cfg_ep.demand_vph.values()))),
                       "epsilon": agent.epsilon, "decisions": res["decisions"],
                       "total_reward": res["total_reward"],
                       "mean_reward": res["total_reward"] / max(res["decisions"], 1),
                       "mean_loss": res["mean_loss"], "mean_q": res["mean_q"],
                       "avg_wait_s": s["avg_waiting_time_s"], "avg_queue": s["avg_queue_length"],
                       "throughput_vph": s["throughput_veh_per_hour"], "phase_switches": s["phase_switches"],
                       "buffer": len(agent.buffer), "wall_s": time.time() - t0,
                       "val_dqn_wait_s": "", "val_fixed_wait_s": ""}
                if (ep + 1) % a.val_every == 0 or ep + 1 == a.episodes:
                    last_val_dqn = validate(val_env, agent, val_seeds)
                    row["val_dqn_wait_s"], row["val_fixed_wait_s"] = last_val_dqn, last_val_fixed
                    if last_val_dqn < best_val:
                        best_val = last_val_dqn
                        agent.save(BEST_CKPT, meta_now(ep + 1))
                        print(f"   new best on validation: {best_val:.1f}s -> {BEST_CKPT.name}")
                writer.writerow(row)
                fh.flush()
                print(f"ep {ep + 1:4d}/{a.episodes} | eps {agent.epsilon:.2f} | R {res['total_reward']:9.1f} | "
                      f"loss {res['mean_loss']:.4f} | wait {s['avg_waiting_time_s']:6.1f}s | "
                      f"queue {s['avg_queue_length']:5.1f} | {row['wall_s']:.0f}s")
                if (ep + 1) % a.save_every == 0:
                    agent.save(LAST_CKPT, meta_now(ep + 1))
            agent.save(LAST_CKPT, meta_now(agent.episodes))
    except KeyboardInterrupt:
        print("\nInterrupted - saving last.pt ...")
        agent.save(LAST_CKPT, meta_now(agent.episodes))
        return 130
    except (SumoNotFoundError, SimulationError, FileNotFoundError, FloatingPointError, ValueError) as exc:
        print(f"ERROR: {exc}")
        try:
            agent.save(LAST_CKPT, meta_now(agent.episodes))
        except Exception:
            pass
        return 1
    finally:
        env.close()
        val_env.close()

    summary = {"episodes_completed": agent.episodes, "env_steps": agent.env_steps,
               "learn_steps": agent.learn_steps, "best_val_wait_s": best_val,
               "val_fixed_wait_s": last_val_fixed, "wall_clock_s": time.time() - t_start,
               "checkpoints": {"best": str(BEST_CKPT) if BEST_CKPT.exists() else None, "last": str(LAST_CKPT)},
               "note": "Validation numbers are on validation seeds only. Run "
                       "`python -m reinforcement_learning.evaluate` for held-out test seeds."}
    TRAIN_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nTraining run finished ({agent.episodes} episodes). Log: {TRAIN_LOG}")
    print("This only means the loop completed. Whether the policy is any good is decided by "
          "`python -m reinforcement_learning.evaluate` on held-out seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())