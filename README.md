# RSRNav Offline Experiments

Testing whether RSRNav's spatial correlation mechanism retains its advantages when trained offline from limited data.

## Setup

### 1. Install Habitat

```bash
# Create conda env
conda create -n rsrnav python=3.9 cmake=3.14 -y
conda activate rsrnav

# Install habitat-sim (headless, no GUI needed)
conda install habitat-sim headless -c conda-forge -c aihabitat -y

# Install habitat-lab
pip install habitat-lab

# Install project dependencies
cd rsrnav-exp
pip install -r requirements.txt
```

### 2. Gibson Dataset

**Step A — Download Gibson scenes** from the Stanford GibsonEnv repo.
Agree to the terms of use at: https://github.com/StanfordVL/GibsonEnv#database

You want the **Habitat-compatible** version (the `.glb` + `.navmesh` files).

**Step B — Download the scene dataset config.** This JSON file is required
by Habitat to correctly load Gibson scenes:

```bash
# Download and place in the gibson root directory
cd rsrnav-exp
mkdir -p data/scene_datasets/gibson
wget -O data/scene_datasets/gibson/gibson_semantic.scene_dataset_config.json \
  http://dl.fbaipublicfiles.com/habitat/gibson/config_v1/gibson_semantic.scene_dataset_config.json
```

**Step C — Place the scene files.** Extract the Gibson download so the
directory looks like this:

```
rsrnav-exp/
└── data/
    └── scene_datasets/
        └── gibson/
            ├── gibson_semantic.scene_dataset_config.json   ← from Step B
            ├── Adrian.glb
            ├── Adrian.navmesh
            ├── Albertville.glb
            ├── Albertville.navmesh
            └── ...  (72 train + 14 val scenes)
```

**Step D — Download ImageNav episode dataset.** These define start/goal
pairs for the navigation task:

```bash
# Using habitat-sim's download utility
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

For the full ImageNav episodes on Gibson, download from the habitat-lab
data page or generate them with habitat-lab's episode generator. The
expected layout:

```
rsrnav-exp/
└── data/
    └── datasets/
        └── imagenav/
            └── gibson/
                └── v2/
                    ├── train/
                    │   ├── train.json.gz
                    │   └── content/   # per-scene episode files
                    └── val/
                        └── val.json.gz
```

### 3. Habitat Config

Create the task config:

```yaml
# configs/habitat_imagenav.yaml
habitat:
  env_task: "GymHabitatEnv"
  task:
    type: "Nav-v0"
    success_distance: 1.0
    measurements:
      - "DistanceToGoal"
      - "Success"
      - "SPL"
    actions:
      move_forward:
        type: "MoveForwardAction"
        forward_step_size: 0.25
      turn_left:
        type: "TurnLeftAction"
        turn_angle: 30
      turn_right:
        type: "TurnRightAction"
        turn_angle: 30
      stop:
        type: "StopAction"
    goal:
      type: "ImageGoal"
  simulator:
    type: "Sim-v0"
    scene_dataset_config_file: "data/scene_datasets/gibson/gibson_semantic.scene_dataset_config.json"
    habitat_sim_v0:
      gpu_device_id: 0
    agents:
      main_agent:
        sim_sensors:
          rgb_sensor:
            type: "HabitatSimRGBSensor"
            height: 128
            width: 128
            hfov: 90
  dataset:
    type: "ImageNav-v1"
    split: "train"
    data_path: "data/datasets/imagenav/gibson/v2/{split}/{split}.json.gz"
    scenes_dir: "data/scene_datasets"
```

## Running

### Step 1: Train online reference (~24h)

Edit `training/train_online.py`:
```python
HABITAT_CONFIG = "configs/habitat_imagenav.yaml"
TOTAL_STEPS = 25_000_000
```

```bash
python -m training.train_online
```

### Step 2: Collect offline datasets (~10h)

Edit `data/collector.py`:
```python
HABITAT_CONFIG = "configs/habitat_imagenav.yaml"
CHECKPOINT = "checkpoints/online_reference/final_model.pt"
```

Run for each policy type — edit `POLICY_TYPE` and `OUTPUT_DIR` each time:
```python
# Run 1: Expert data
POLICY_TYPE = "expert"
OUTPUT_DIR = "datasets/raw/expert"
EPISODES_PER_SCENE = 9000

# Run 2: Noisy data
POLICY_TYPE = "noisy"
OUTPUT_DIR = "datasets/raw/noisy"
EPISODES_PER_SCENE = 1000

# Run 3: Mixed data
POLICY_TYPE = "mixed"
OUTPUT_DIR = "datasets/raw/mixed"
EPISODES_PER_SCENE = 1000

# Run 4: Random data (no checkpoint needed)
POLICY_TYPE = "random"
OUTPUT_DIR = "datasets/raw/random"
EPISODES_PER_SCENE = 9000
```

```bash
python -m data.collector  # run once per config above
```

### Step 3: Generate degraded datasets

Edit `data/degradation.py` for each variant:
```python
INPUT = "datasets/raw/expert/episodes.hdf5"
OUTPUT = "datasets/combined-mild/episodes.hdf5"
PRESET = "mild"
```

```bash
python -m data.degradation  # run once per preset: mild, moderate, severe
```

### Step 4: Train offline models (~3-4h each)

Edit `training/train_iql.py`:
```python
HABITAT_CONFIG = "configs/habitat_imagenav.yaml"
DATASET = "datasets/raw/expert/episodes.hdf5"
CORRELATION_TYPE = "direction_aware"  # change per experiment
SAVE_DIR = "checkpoints/exp1_da_ideal"
LOG_DIR = "runs/exp1_da_ideal"
```

```bash
python -m training.train_iql  # run once per config
```

### Step 5: Evaluate

Edit `eval/evaluate.py`:
```python
CHECKPOINT = "checkpoints/exp1_da_ideal/best_model.pt"
HABITAT_CONFIG = "configs/habitat_imagenav.yaml"
```

```bash
python -m eval.evaluate
```

### Step 6: Aggregate results

```bash
python -m eval.aggregate_results
```

## Quick Verification (no Habitat)

Test the full pipeline with dummy environments:
```bash
python -m data.collector        # uses dummy env when HABITAT_CONFIG = None
python -m training.train_iql    # trains on collected dummy data
python -m eval.aggregate_results
```

## Directory Structure

```
rsrnav-exp/
├── configs/
│   └── habitat_imagenav.yaml
├── data/
│   ├── scene_datasets/gibson/    ← Gibson .glb files go here
│   ├── datasets/imagenav/        ← Episode JSON files go here
│   ├── collector.py
│   ├── degradation.py
│   └── offline_dataset.py
├── datasets/                     ← Generated HDF5 datasets (gitignored)
├── checkpoints/                  ← Saved models (gitignored)
├── results/                      ← Eval JSONs (gitignored)
├── models/
├── envs/
├── training/
├── eval/
└── scripts/
```
