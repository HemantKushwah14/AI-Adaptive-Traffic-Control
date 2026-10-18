"""Paths and seed ranges for DQN training. Seed ranges are disjoint to prevent leakage."""
from __future__ import annotations

from config.settings import MODELS_DIR, OUTPUTS_DIR

DQN_DIR = MODELS_DIR / "dqn"            # checkpoints
DQN_OUT_DIR = OUTPUTS_DIR / "dqn"       # logs, evaluation CSVs
BEST_CKPT = DQN_DIR / "best.pt"         # best on VALIDATION seeds
LAST_CKPT = DQN_DIR / "last.pt"         # most recent (for --resume)
TRAIN_LOG = DQN_OUT_DIR / "training_log.csv"
TRAIN_SUMMARY = DQN_OUT_DIR / "training_summary.json"

TRAIN_SEED_BASE = 1000      # episode i uses 1000 + i      (i < 8000)
VAL_SEED_BASE = 9000        # validation episodes
TEST_SEED_BASE = 20000      # final evaluation / comparison
MAX_TRAIN_EPISODES = 8000   # keeps train seeds below VAL_SEED_BASE