#!/usr/bin/env bash
#
# Run the full offline RL experiment sweep.
#
# Usage:
#   chmod +x offline/run_experiments.sh
#   ./offline/run_experiments.sh
#
set -e

DEVICE="cuda"
EXPERT_EPISODES=1000
RANDOM_EPISODES=500
IQL_STEPS=200000
BC_STEPS=100000

echo "============================================"
echo " RSRNav Offline RL Experiment Sweep"
echo "============================================"

# ─── Step 1: Collect data ───
echo ""
echo "=== Phase 1: Data Collection ==="

echo "[1/3] Collecting expert data..."
python offline/collect.py --policy expert --episodes $EXPERT_EPISODES --device $DEVICE

echo "[2/3] Collecting noisy expert data..."
python offline/collect.py --policy noisy --episodes $EXPERT_EPISODES --device $DEVICE

echo "[3/3] Collecting random data..."
python offline/collect.py --policy random --episodes $RANDOM_EPISODES --device $DEVICE

EXPERT_DATA="offline_data/raw/expert_${EXPERT_EPISODES}ep/episodes.hdf5"
RANDOM_DATA="offline_data/raw/random_${RANDOM_EPISODES}ep/episodes.hdf5"

# ─── Step 2: Create degraded datasets ───
echo ""
echo "=== Phase 2: Data Degradation ==="
PRESETS="vol25 vol50 vol75 trunc25 trunc50 obs_noise_low obs_noise_high obs_dropout_10 obs_dropout_30"

for preset in $PRESETS; do
    echo "  Degrading: $preset"
    python offline/degrade.py --input $EXPERT_DATA --preset $preset
done

echo "  Degrading: quality_mixed"
python offline/degrade.py --input $EXPERT_DATA --preset quality_mixed --random-input $RANDOM_DATA

# ─── Step 3: Train on full data ───
echo ""
echo "=== Phase 3: Training on Full Data ==="

echo "[IQL] Full expert data..."
python offline/train_iql.py --dataset $EXPERT_DATA \
    --total-steps $IQL_STEPS --device $DEVICE \
    --save-dir checkpoints/iql/full --run-name iql_full

echo "[BC] Full expert data..."
python offline/train_bc.py --dataset $EXPERT_DATA \
    --total-steps $BC_STEPS --device $DEVICE \
    --save-dir checkpoints/bc/full --run-name bc_full

# ─── Step 4: Train on degraded data ───
echo ""
echo "=== Phase 4: Training on Degraded Data ==="

for preset in $PRESETS quality_mixed; do
    DEGRADED="offline_data/degraded/${preset}/episodes.hdf5"
    if [ -f "$DEGRADED" ]; then
        echo "[IQL] $preset..."
        python offline/train_iql.py --dataset $DEGRADED \
            --total-steps $IQL_STEPS --device $DEVICE \
            --save-dir "checkpoints/iql/${preset}" \
            --run-name "iql_${preset}"

        echo "[BC] $preset..."
        python offline/train_bc.py --dataset $DEGRADED \
            --total-steps $BC_STEPS --device $DEVICE \
            --save-dir "checkpoints/bc/${preset}" \
            --run-name "bc_${preset}"
    fi
done

# ─── Step 5: Evaluate all ───
echo ""
echo "=== Phase 5: Evaluation ==="

# Expert baseline
echo "Evaluating expert baseline..."
python offline/evaluate.py --method expert --tag full_data

# Random baseline
echo "Evaluating random baseline..."
python offline/evaluate.py --method random --tag full_data

# Full data models
IQL_CKPT="checkpoints/iql/full/ckpt_$(printf '%06d' $IQL_STEPS).pt"
BC_CKPT="checkpoints/bc/full/ckpt_$(printf '%06d' $BC_STEPS).pt"

echo "Evaluating IQL (full data)..."
python offline/evaluate.py --method iql --checkpoint $IQL_CKPT --tag full_data

echo "Evaluating BC (full data)..."
python offline/evaluate.py --method bc --checkpoint $BC_CKPT --tag full_data

# Degraded data models
for preset in $PRESETS quality_mixed; do
    IQL_CKPT="checkpoints/iql/${preset}/ckpt_$(printf '%06d' $IQL_STEPS).pt"
    BC_CKPT="checkpoints/bc/${preset}/ckpt_$(printf '%06d' $BC_STEPS).pt"

    if [ -f "$IQL_CKPT" ]; then
        echo "Evaluating IQL ($preset)..."
        python offline/evaluate.py --method iql --checkpoint $IQL_CKPT --tag $preset
    fi
    if [ -f "$BC_CKPT" ]; then
        echo "Evaluating BC ($preset)..."
        python offline/evaluate.py --method bc --checkpoint $BC_CKPT --tag $preset
    fi
done

# ─── Step 6: Aggregate ───
echo ""
echo "=== Phase 6: Aggregation ==="
python offline/aggregate.py

echo ""
echo "============================================"
echo " All experiments complete!"
echo "============================================"
