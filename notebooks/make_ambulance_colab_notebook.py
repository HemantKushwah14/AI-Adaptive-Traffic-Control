"""Generate notebooks/train_ambulance_colab.ipynb.  Run: python notebooks\\make_ambulance_colab_notebook.py"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "train_ambulance_colab.ipynb"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# Ambulance detector fine-tuning - Colab (T4 GPU)
Upload to `MyDrive/traffic_ai/`: `traffic_ai_ambulance.zip` (config, utils, dataset) and
`ambulance_raw.zip` (YOUR downloaded YOLO-format ambulance dataset; check its licence).
All metrics below are computed by these cells on the held-out TEST split.
"""),
    code("""
import torch
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
print(torch.cuda.get_device_name(0))
!pip -q install ultralytics==8.3.0 PyYAML tqdm python-dotenv Pillow pandas
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, sys
D = Path('/content/drive/MyDrive/traffic_ai')
!rm -rf /content/traffic_ai_project && mkdir -p /content/traffic_ai_project/data/ambulance_raw
!unzip -q -o "{D}/traffic_ai_ambulance.zip" -d /content/traffic_ai_project
!unzip -q -o "{D}/ambulance_raw.zip" -d /content/traffic_ai_project/data/ambulance_raw
os.chdir('/content/traffic_ai_project'); sys.path.insert(0, os.getcwd())
"""),
    md("## 1. Inspect classes, then prepare a clean single-class dataset"),
    code("""
!python -m dataset.prepare_ambulance_dataset --list-classes
"""),
    code("""
AMB_CLASSES = "ambulance"   # edit to match the class names printed above (comma-separated)
!python -m dataset.prepare_ambulance_dataset --ambulance-classes "{AMB_CLASSES}" --overwrite
!cat data/ambulance_yolo/prepare_report.json
"""),
    md("## 2. Train"),
    code("""
from ultralytics import YOLO
import json, shutil
DATA = '/content/traffic_ai_project/data/ambulance_yolo/data.yaml'
RUNS = D / 'runs'
EPOCHS, IMGSZ, BATCH = 50, 640, 16
model = YOLO('yolov8n.pt')
model.train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, project=str(RUNS),
            name='ambulance_detector', exist_ok=True, seed=42, deterministic=True, patience=15, workers=2)
BEST = Path(model.trainer.best)
"""),
    md("## 3. Evaluate on the held-out TEST split"),
    code("""
best = YOLO(str(BEST))
m = best.val(data=DATA, split='test', imgsz=IMGSZ, batch=BATCH, device=0, project=str(RUNS),
             name='ambulance_test_eval', exist_ok=True)
report = {'split': 'test', 'map50': float(m.box.map50), 'map50_95': float(m.box.map),
          'precision': float(m.box.mp), 'recall': float(m.box.mr), 'epochs': EPOCHS, 'imgsz': IMGSZ,
          'caveat': 'Valid only for data similar to the training set; see Part 6 limitations.'}
print(json.dumps(report, indent=2))
(D / 'ambulance_detector_test_metrics.json').write_text(json.dumps(report, indent=2))
"""),
    md("## 4. Save weights\nCopy `ambulance_detector.pt` to `traffic_ai_project\\\\models\\\\ambulance_detector.pt` on your PC."),
    code("""
shutil.copy2(BEST, D / 'ambulance_detector.pt')
shutil.copy2('data/ambulance_yolo/prepare_report.json', D / 'ambulance_prepare_report.json')
print('Saved to', D)
"""),
]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")