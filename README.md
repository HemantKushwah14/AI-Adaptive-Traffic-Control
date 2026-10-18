# Adaptive Traffic Signal Optimization using CV, DQN and Deep Reinforcement Learning

M.Tech project. A four-way intersection is simulated in **Eclipse SUMO**. A **Deep Q-Network** chooses the next
signal phase and the choice is applied to the real SUMO traffic light through **TraCI**, with enforced yellow and
all-red transitions. A **YOLO** (CNN) module detects and counts vehicles in images and videos. An
**emergency-priority controller** pre-empts the signal for ambulances. A **Streamlit** dashboard shows everything.

## 1. What is real, simulated, and what needs training

| Component | Kind | Needs from you |
|---|---|---|
| SUMO network, routes, signals, TraCI control, metrics | Working implementation | SUMO (installed by pip) |
| Fixed-time baseline | Working implementation | nothing |
| DQN (replay, target net, epsilon-greedy, Bellman) | Working implementation | **Training run** (`reinforcement_learning.train`) before it means anything |
| Vehicle detector | Pretrained COCO YOLO by default | Optional **Colab fine-tuning** on converted UA-DETRAC |
| Lane/approach assignment from camera images | Working, **needs calibration** | `scripts/calibrate_regions.py` on your camera frame |
| Ambulance detector (vision) | Code complete, **no model shipped** | An ambulance dataset you download + Colab training |
| Ambulance priority in SUMO | Working; ambulances and their detection are **simulated** (ground truth via TraCI) | nothing |
| Vision stream -> live signal control | **Not implemented**: the DQN is trained and run on SUMO-measured state | out of scope; see limitations |

## 2. Architecture

```text
Images / video -> YOLO detection -> counting -> region-based density/queue (Page 3, standalone analysis)

SUMO (traffic) --TraCI--> state (17-D) --> DQN --> next phase --> SignalManager --> SUMO signal
        ^                                                    (green -> yellow -> all-red -> green)
        |                                                                      |
        +---- metrics (waiting, queue, throughput, delay) <--------------------+
Ambulance (simulated or vision source) -> EmergencyController(base = DQN | fixed) -> preempt -> clear -> restore
Dashboard (7 pages) reads one background simulation session; long jobs run in background threads.
```

State (17): density(4), queue(4), waiting(4), one-hot phase(4), elapsed green(1). Actions: the 4 conflict-free phases
(NS-through, NS-left, EW-through, EW-left). Reward: -waiting - queue - fairness - switching + throughput.
Bellman target: `y = r + gamma * max_{a' legal} Q_target(s', a')`; loss: Huber.

## 3. Layout

```text
app.py                  Streamlit entry point          config/        constants, settings, CV/RL/emergency config
simulation/             SUMO build, routes, TraCI, signal state machine, fixed/adaptive/emergency controllers
reinforcement_learning/ network, replay buffer, reward, action mask, agent, train, evaluate
computer_vision/        detector, counter/tracker, lane analyzer, video pipeline, ambulance detector
dataset/                download, XML->YOLO conversion, validation, ambulance dataset prep
evaluation/             metrics, comparison, reports          dashboard/   pages, session, jobs, charts, theme
scripts/                runnable entry points                 tests/       pytest suite
notebooks/              generators for the Colab notebooks    sumo_config/ generated SUMO files
data/ models/ outputs/  git-ignored: datasets, weights, results
```

## 4. Setup (Windows 11, PowerShell, Python 3.11)

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\check_environment.py        # every line must say [ OK ]
python -m simulation.build_network --force
python scripts\run_all_checks.py --quick   # fast confidence check
```

## 5. Workflows

**Simulation:** `python scripts\run_fixed_time.py --duration 600 [--gui]`

**Vehicle data and detector** (needs internet):
```powershell
python -m dataset.download_dataset --list
python -m dataset.download_dataset --max-sequences 30 --frame-stride 10
python -m dataset.validate_dataset --raw      # prints the vehicle classes that REALLY exist
python -m dataset.prepare_dataset --overwrite
python -m dataset.validate_dataset --yolo
python scripts\run_detection.py --image <frame.jpg>
python notebooks\make_colab_notebook.py       # then follow the notebook on a Colab T4 GPU
```
Place the trained `vehicle_detector.pt` in `models\`; copy `vehicle_detector_test_metrics.json` to `outputs\vision\`.

**DQN:**
```powershell
python -m reinforcement_learning.dqn_agent --selftest
python -m reinforcement_learning.train --episodes 80 --duration 900     # resumable with --resume
python -m reinforcement_learning.evaluate --seeds 3
```

**Comparison (identical demand and seeds, route files hash-verified):**
```powershell
python scripts\run_comparison.py --seeds 5 --duration 900 --demand 500
python scripts\run_comparison.py --seeds 3 --fixed-greens 30,12,30,12   # test a stronger baseline too
```

**Emergency:** `python scripts\run_emergency_demo.py [--base dqn]`

**Dashboard:** `streamlit run app.py`

## 6. Testing

```powershell
python -m pytest -q                         # unit + SUMO integration (integration auto-skips without SUMO)
python -m pytest -q -m "not integration"    # unit tests only
python scripts\run_all_checks.py            # everything, with a JSON report in outputs\
python scripts\run_all_checks.py --full     # + DQN session, comparison, emergency demo (needs a checkpoint)
python scripts\generate_results_summary.py --out docs\RESULTS_SUMMARY.md
```

## 7. Demonstration script (about 15 minutes)

Prepare **before** the evaluation: train the DQN, run `run_comparison.py --seeds 5`, run the emergency demo, run
`run_all_checks.py` and `generate_results_summary.py`. Comparisons take minutes, so show saved results and offer a live
short run only if asked.

1. **Problem and architecture (2 min)**: fixed-time signals waste green time; show the diagram above.
2. **Test evidence (1 min)**: show `outputs\all_checks_report.json`.
3. **Page 2, live simulation (3 min)**: run fixed-time, then reset and run the DQN with the same seed. Point out
   yellow/all-red, queues, and that the picture is drawn from TraCI data.
4. **Page 4, DQN (2 min)**: Q-values, legal actions, state table, training curves. State that Q-values are not a causal explanation.
5. **Page 3, detection (2 min)**: upload an image, show boxes and class counts, and say which model mode is active.
6. **Page 5, emergency (2 min)**: dispatch an ambulance, show preemption, clearance and restoration; say it is simulated.
7. **Page 6, comparison (2 min)**: present measured numbers including seeds where the DQN did *not* win.
8. **Limitations (1 min)**: use section 10.

## 8. Viva preparation (short answers)

- **Why DQN, not tabular Q-learning?** The state is continuous (17-D); a network generalises across traffic states.
- **Why a target network and replay?** Replay breaks temporal correlation; a slowly updated target stabilises the bootstrap.
- **Why mask actions?** Min-green and max-green rules make some phases illegal; the agent never bootstraps over them.
- **Why is `done` not terminal?** An episode ends at a time limit, not a natural end state, so targets keep bootstrapping.
- **How do you prevent unsafe signals?** Only 4 conflict-free phases exist, and every change goes green -> yellow -> all-red -> green.
- **Is the comparison fair?** Same demand, duration and seeds; route files are hashed; seeds are held out from training.
- **What if the DQN does not beat fixed-time?** That is a valid result; report it and discuss reward/baseline/training length.
- **What does the YOLO network do?** Backbone (conv + C2f) extracts features, PAN-FPN neck fuses scales, decoupled anchor-free head predicts boxes and classes; NMS removes duplicates.
- **Why split by sequence?** Neighbouring frames are near-duplicates; a frame-level split leaks test data.
- **Why is ambulance detection hard?** Look-alikes, livery variation, night/rain, occlusion, distance, domain shift; hence multi-frame confirmation.

## 9. Reproducibility

Seeds: training 1000+episode, validation 9000+i, test 20000+i. The same `SimulationConfig` and seed give an identical route
file and identical SUMO results. Checkpoints store training metadata; the UI warns when run settings differ from it.

## 10. Limitations

- The DQN trains on SUMO-measured state, not on camera images; vision is not wired into live control.
- A trained DQN is evidenced only for conditions near its training range (about 300-800 veh/h per approach, 10-60 s green).
- The default fixed-time plan is hand-chosen, not optimised; a stronger baseline can shrink the measured gain.
- No statistical significance test; read the per-seed consistency column.
- UA-DETRAC has no motorcycle or truck labels; fine-tuning cannot teach them. The mapping `others -> truck`, `van -> car` is an assumption.
- Camera regions must be calibrated per camera; the default template is a demo and is flagged everywhere.
- Ambulances and their detection inside SUMO are simulated. The vision ambulance detector needs your own dataset and training.
- The restoration after an emergency continues from the emergency phase; the interrupted phase is not replayed.
- Single intersection, passenger-style vehicle types, no pedestrians.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named ...` | Activate `.venv`, then `pip install -r requirements.txt` |
| SUMO not found | `pip install eclipse-sumo traci sumolib`, then `python scripts\check_environment.py` |
| Stale `sumo.exe` | `taskkill /F /IM sumo.exe` |
| Stale network after edits | `python -m simulation.build_network --force` |
| Page shows no DQN | Train first (section 5) or choose Fixed-time |
| Detector cannot download weights | Download `yolov8n.pt` manually into `models\` |

## 12. GitHub

```powershell
git init -b main
git add .
git status        # confirm .venv, data, outputs and weights are NOT listed
git commit -m "Initial commit"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
Datasets are not redistributed; check the licence of UA-DETRAC, its Hugging Face mirror, and your ambulance dataset.

## 13. References (verify formats before citing)

Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015. Lopez et al., "Microscopic Traffic
Simulation using SUMO", IEEE ITSC 2018. Wen et al., "UA-DETRAC: A New Benchmark and Protocol for Multi-Object
Detection and Tracking", 2015. Ultralytics YOLOv8 documentation.

## 14. Results

Not written by hand. Generate `docs\RESULTS_SUMMARY.md` with `scripts\generate_results_summary.py` after your runs.
