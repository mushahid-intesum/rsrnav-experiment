# RSRNav Offline RL Pipeline

Single-GPU offline RL experiments on top of the official [RSRNav](https://github.com/qinzheng2000/RSRNav) codebase.

## Setup

```bash
# 1. Create conda env (Python 3.9, habitat-sim, PyTorch)
chmod +x offline/setup_env.sh
./offline/setup_env.sh

# 2. Activate and verify
conda activate rsrnav
python offline/inspect_checkpoint.py
```

**Requirements**: Gibson scene data at `data/scene_datasets/gibson/` and ImageNav episodes at `data/datasets/imagenav/gibson/v2/`.

## Quick Start

```bash
# Run the full experiment sweep (collect → degrade → train → eval → report)
chmod +x offline/run_experiments.sh
./offline/run_experiments.sh
```

## Individual Steps

### 1. Collect data
```bash
python offline/collect.py --policy expert --episodes 1000
python offline/collect.py --policy random --episodes 500
python offline/collect.py --policy noisy  --episodes 1000
```

### 2. Degrade data
```bash
python offline/degrade.py --input offline_data/raw/expert_1000ep/episodes.hdf5 --preset vol25
# Presets: vol25, vol50, vol75, trunc25, trunc50, obs_noise_low, obs_noise_high,
#          obs_dropout_10, obs_dropout_30, quality_mixed
```

### 3. Train offline
```bash
# IQL
python offline/train_iql.py --dataset offline_data/raw/expert_1000ep/episodes.hdf5

# BC
python offline/train_bc.py --dataset offline_data/raw/expert_1000ep/episodes.hdf5

# With degraded data
python offline/train_iql.py --dataset offline_data/degraded/vol25/episodes.hdf5
```

### 4. Evaluate
```bash
python offline/evaluate.py --method iql --checkpoint checkpoints/iql/full/ckpt_200000.pt --tag full_data
python offline/evaluate.py --method bc  --checkpoint checkpoints/bc/full/ckpt_100000.pt  --tag full_data
python offline/evaluate.py --method expert --tag full_data
```

### 5. Aggregate results
```bash
python offline/aggregate.py
# Outputs: offline_data/results/results.md, results_table.tex, plots/
```

## Architecture

The standalone model (`offline/model.py`) reconstructs the official RSRNav architecture:
- **Encoder**: ResNet-9 (GroupNorm, SpaceToDepth stem) — weight-shared for obs + goal
- **Correlation**: RAFT-style CorrBlock (3 pyramid levels, radius=1, bilinear sampling)
- **MotionEncoder**: Conv 27→64→64→32
- **Policy**: GRU (2 layers, hidden=128) + Linear actor/critic heads
- **Total**: 1,561,157 parameters

Backbone is frozen during offline training. Only the IQL (Q/V/actor) or BC (policy head) networks are trained.

## File Structure

| File | Purpose |
|---|---|
| `configs.py` | All constants, paths, hyperparameters |
| `model.py` | Standalone RSRNav model (no habitat dependency) |
| `dataset.py` | HDF5 episode → PyTorch DataLoader |
| `collect.py` | Roll out policy in Habitat → HDF5 |
| `degrade.py` | Apply data degradation presets |
| `train_iql.py` | IQL offline trainer (single GPU) |
| `train_bc.py` | BC baseline trainer (single GPU) |
| `evaluate.py` | Evaluate in Habitat → JSON metrics |
| `aggregate.py` | Aggregate results → tables + plots |
| `run_experiments.sh` | Full sweep automation |
| `setup_env.sh` | Conda environment setup |
| `inspect_checkpoint.py` | Verify official checkpoint loads |
