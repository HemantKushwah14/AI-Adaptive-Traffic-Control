"""Generate notebooks/train_dqn_colab.ipynb.   Run:  python notebooks\\make_dqn_colab_notebook.py"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "train_dqn_colab.ipynb"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# DQN traffic-signal training (SUMO + PyTorch) - Colab
The DQN is a tiny MLP; the bottleneck is SUMO/TraCI on the CPU, so a **GPU is NOT required** here.
Colab is useful because it runs long jobs while your PC is free. Use the default CPU runtime.
Upload `traffic_ai_dqn.zip` (config, utils, simulation, evaluation, reinforcement_learning) to `MyDrive/traffic_ai/`.
No results are pre-filled: everything printed below is measured by the cells.
"""),
    code("""
!pip -q install eclipse-sumo==1.21.0 traci==1.21.0 sumolib==1.21.0 python-dotenv pandas
import torch, shutil
print("torch", torch.__version__, "| sumo on PATH:", shutil.which("sumo"), shutil.which("netconvert"))
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, sys
DRIVE_DIR = Path('/content/drive/MyDrive/traffic_ai')
ZIP = DRIVE_DIR / 'traffic_ai_dqn.zip'
assert ZIP.exists(), f"Upload {ZIP.name} to {DRIVE_DIR} first"
!rm -rf /content/traffic_ai_project && mkdir -p /content/traffic_ai_project
!unzip -q -o "{ZIP}" -d /content/traffic_ai_project
os.chdir('/content/traffic_ai_project'); sys.path.insert(0, os.getcwd())
print(sorted(os.listdir('.')))
"""),
    md("## 1. Build the SUMO network and run the learning-rule self-test"),
    code("""
!python -m simulation.build_network --force
!python -m reinforcement_learning.dqn_agent --selftest
!python scripts/run_fixed_time.py --duration 300 2>/dev/null || echo "(scripts/ not in zip: skipped)"
"""),
    md("## 2. Train (edit the numbers; start small to measure time per episode)"),
    code("""
EPISODES, DURATION = 80, 900
!python -m reinforcement_learning.train --episodes {EPISODES} --duration {DURATION}
"""),
    md("## 3. Evaluate on HELD-OUT test seeds against the fixed-time baseline"),
    code("""
!python -m reinforcement_learning.evaluate --seeds 3 --duration {DURATION}
"""),
    code("""
import pandas as pd
log = pd.read_csv('outputs/dqn/training_log.csv')
print(log[['episode','epsilon','total_reward','mean_loss','avg_wait_s','val_dqn_wait_s','val_fixed_wait_s']].tail(15))
"""),
    md("## 4. Save to Drive, then copy `models/dqn/best.pt` to your PC's `traffic_ai_project\\\\models\\\\dqn\\\\`"),
    code("""
import shutil
dst = DRIVE_DIR / 'dqn_run'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree('models/dqn', dst / 'models_dqn')
shutil.copytree('outputs/dqn', dst / 'outputs_dqn')
print('Saved to', dst)
"""),
]

nb = {"cells": CELLS,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")