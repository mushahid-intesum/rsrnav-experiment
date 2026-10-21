#!/bin/bash
# =============================================================================
# Collect and degrade offline datasets for RSRNav experiments.
#
# Prerequisites:
#   - Phase 1 online reference checkpoint at $CHECKPOINT
#   - Habitat environment configured (or use dummy mode without --habitat_config)
#
# Usage:
#   bash scripts/collect_datasets.sh [--dummy]
#
# Pass --dummy to use the dummy environment (no Habitat required).
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CHECKPOINT="${CHECKPOINT:-checkpoints/online_reference/final_model.pt}"
HABITAT_CONFIG="${HABITAT_CONFIG:-}"
BASE_OUTPUT="datasets"
SEED=42

# Check for --dummy flag
DUMMY_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--dummy" ]; then
        DUMMY_MODE=true
    fi
done

HAB_FLAG=""
if [ -n "$HABITAT_CONFIG" ] && [ "$DUMMY_MODE" = false ]; then
    HAB_FLAG="--habitat_config $HABITAT_CONFIG"
fi

# Reduce episode count for dummy mode (just testing the pipeline)
if [ "$DUMMY_MODE" = true ]; then
    echo "=== DUMMY MODE: Using small episode counts for pipeline testing ==="
    EP_FULL=100
    EP_1K=20
    EP_250=10
    EP_50=5
else
    EP_FULL=9000
    EP_1K=1000
    EP_250=250
    EP_50=50
fi

echo "============================================"
echo "  RSRNav Offline Dataset Collection"
echo "============================================"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output dir: $BASE_OUTPUT"
echo ""

# ---- Step 1: Collect raw datasets with different behavior policies ----

echo "=== Step 1: Collecting with behavior policies ==="

# Expert rollouts (full volume)
echo "[1/4] Expert policy ($EP_FULL episodes)..."
python3 -m data.collector \
    --policy expert \
    --checkpoint "$CHECKPOINT" \
    --episodes_per_scene $EP_FULL \
    --output_dir "$BASE_OUTPUT/raw/expert" \
    --seed $SEED \
    $HAB_FLAG || echo "  (Skipped: checkpoint not found)"

# Noisy policy
echo "[2/4] Noisy policy ($EP_1K episodes)..."
python3 -m data.collector \
    --policy noisy \
    --checkpoint "$CHECKPOINT" \
    --epsilon 0.25 \
    --episodes_per_scene $EP_1K \
    --output_dir "$BASE_OUTPUT/raw/noisy" \
    --seed $SEED \
    $HAB_FLAG || echo "  (Skipped: checkpoint not found)"

# Mixed policy
echo "[3/4] Mixed policy ($EP_1K episodes)..."
python3 -m data.collector \
    --policy mixed \
    --checkpoint "$CHECKPOINT" \
    --episodes_per_scene $EP_1K \
    --output_dir "$BASE_OUTPUT/raw/mixed" \
    --seed $SEED \
    $HAB_FLAG || echo "  (Skipped: checkpoint not found)"

# Random policy (no checkpoint needed)
echo "[4/4] Random policy ($EP_FULL episodes)..."
python3 -m data.collector \
    --policy random \
    --episodes_per_scene $EP_FULL \
    --output_dir "$BASE_OUTPUT/raw/random" \
    --seed $SEED \
    $HAB_FLAG

echo ""
echo "=== Step 2: Applying degradation presets ==="

# ---- Step 2: Generate degraded variants from expert data ----

SOURCE="$BASE_OUTPUT/raw/expert/episodes.hdf5"

if [ -f "$SOURCE" ]; then
    # Volume variants (Experiment 2)
    for N in $EP_1K $EP_250 $EP_50; do
        echo "Volume reduction: $N episodes..."
        python3 -m data.degradation \
            --input "$SOURCE" \
            --output "$BASE_OUTPUT/vol-${N}/episodes.hdf5" \
            --max_episodes $N \
            --seed $SEED --stats
    done

    # Combined realistic presets (Experiment 4)
    echo "Preset: mild..."
    python3 -m data.degradation \
        --input "$SOURCE" \
        --output "$BASE_OUTPUT/combined-mild/episodes.hdf5" \
        --preset mild --seed $SEED --stats

    echo "Preset: moderate..."
    python3 -m data.degradation \
        --input "$SOURCE" \
        --output "$BASE_OUTPUT/combined-moderate/episodes.hdf5" \
        --preset moderate --seed $SEED --stats

    echo "Preset: severe..."
    python3 -m data.degradation \
        --input "$SOURCE" \
        --output "$BASE_OUTPUT/combined-severe/episodes.hdf5" \
        --preset severe --seed $SEED --stats

else
    echo "WARNING: Expert source not found at $SOURCE"
    echo "  Generating degraded variants from random data instead..."

    SOURCE="$BASE_OUTPUT/raw/random/episodes.hdf5"

    if [ -f "$SOURCE" ]; then
        for preset in mild moderate severe; do
            python3 -m data.degradation \
                --input "$SOURCE" \
                --output "$BASE_OUTPUT/combined-${preset}/episodes.hdf5" \
                --preset $preset --seed $SEED --stats
        done
    fi
fi

echo ""
echo "=== Dataset collection complete ==="
echo ""
echo "Generated datasets:"
find "$BASE_OUTPUT" -name "*.hdf5" -exec ls -lh {} \; 2>/dev/null || true
echo ""
echo "Next: Run experiments with:"
echo "  python3 -m training.train_iql --dataset datasets/<name>/episodes.hdf5"
