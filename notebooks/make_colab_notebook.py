"""Generate notebooks/train_detector_colab.ipynb.   Run:  python notebooks\\make_colab_notebook.py"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "train_detector_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# Vehicle detector fine-tuning (UA-DETRAC subset) - Colab
Before running: **Runtime > Change runtime type > T4 GPU**, and upload `traffic_ai_colab.zip`
(created locally with `Compress-Archive`) to `MyDrive/traffic_ai/`.
Nothing here is pre-trained: every metric printed is computed by the cells below.
"""),
    code("""
import torch, subprocess
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then re-run."
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
"""),
    code("""
!pip -q install ultralytics==8.3.0 huggingface_hub==0.25.2 PyYAML tqdm python-dotenv Pillow pandas
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
DRIVE_DIR = Path('/content/drive/MyDrive/traffic_ai')
PROJECT_ZIP = DRIVE_DIR / 'traffic_ai_colab.zip'
assert PROJECT_ZIP.exists(), f"Upload {PROJECT_ZIP.name} to {DRIVE_DIR} first"
!rm -rf /content/traffic_ai_project && mkdir -p /content/traffic_ai_project
!unzip -q -o "{PROJECT_ZIP}" -d /content/traffic_ai_project
import os, sys
os.chdir('/content/traffic_ai_project'); sys.path.insert(0, os.getcwd())
print(os.listdir('.'))
"""),
    md("## 1. Download a subset of the dataset (edit the numbers if you like)"),
    code("""
MAX_SEQUENCES = 40     # annotated sequences to use (>= 3)
FRAME_STRIDE = 5       # keep every 5th frame
!python -m dataset.download_dataset --list
!python -m dataset.download_dataset --max-sequences {MAX_SEQUENCES} --frame-stride {FRAME_STRIDE}
"""),
    code("""
!python -m dataset.validate_dataset --raw
"""),
    md("## 2. Convert to YOLO format (split by sequence) and validate\nIf you want `van` merged into `car`, add `--merge van=car`."),
    code("""
!python -m dataset.prepare_dataset --overwrite
!python -m dataset.validate_dataset --yolo
"""),
    md("## 3. Train (fine-tune from COCO-pretrained yolov8n)"),
    code("""
from ultralytics import YOLO
import json, shutil
DATA_YAML = Path('/content/traffic_ai_project/data/detrac_yolo/data.yaml')
RUNS_DIR = DRIVE_DIR / 'runs'
EPOCHS, IMGSZ, BATCH = 30, 640, 16

model = YOLO('yolov8n.pt')
results = model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,
                      project=str(RUNS_DIR), name='vehicle_detector', exist_ok=True,
                      seed=42, deterministic=True, patience=10, workers=2)
BEST = Path(model.trainer.best)
print("Best weights:", BEST)
"""),
    md("## 4. Evaluate on the held-out TEST sequences (never used for training or model selection)"),
    code("""
best_model = YOLO(str(BEST))
m = best_model.val(data=str(DATA_YAML), split='test', imgsz=IMGSZ, batch=BATCH, device=0,
                   project=str(RUNS_DIR), name='test_eval', exist_ok=True)
idx = list(m.box.ap_class_index)
report = {
    'split': 'test',
    'map50': float(m.box.map50), 'map50_95': float(m.box.map),
    'precision': float(m.box.mp), 'recall': float(m.box.mr),
    'per_class_map50_95': {best_model.names[int(i)]: float(m.box.maps[int(i)]) for i in idx},
    'classes': {int(k): v for k, v in best_model.names.items()},
    'epochs': EPOCHS, 'imgsz': IMGSZ,
}
print(json.dumps(report, indent=2))
(DRIVE_DIR / 'vehicle_detector_test_metrics.json').write_text(json.dumps(report, indent=2))
"""),
    md("## 5. Save and export"),
    code("""
shutil.copy2(BEST, DRIVE_DIR / 'vehicle_detector.pt')
shutil.copy2('/content/traffic_ai_project/data/detrac_yolo/prepare_report.json',
             DRIVE_DIR / 'prepare_report.json')
try:
    onnx = best_model.export(format='onnx', imgsz=IMGSZ)
    shutil.copy2(onnx, DRIVE_DIR / 'vehicle_detector.onnx')
except Exception as exc:
    print('ONNX export skipped:', exc)
print('Saved to', DRIVE_DIR)
"""),
    md("""
## 6. Use the weights locally
Download `vehicle_detector.pt` from Google Drive and place it at
`traffic_ai_project\\models\\vehicle_detector.pt`. The local detector then switches
automatically from `pretrained-coco` to `finetuned` mode.
Keep `vehicle_detector_test_metrics.json` - these are your real evaluation results.
"""),
]

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")