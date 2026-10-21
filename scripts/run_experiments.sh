#!/bin/bash
# =============================================================================
# Run all RSRNav offline experiments (Exp 1-4 from the plan).
#
# Usage:
#   bash scripts/run_experiments.sh [--fast]     # --fast = Exp 1+4 only
#   bash scripts/run_experiments.sh [--exp N]    # Run specific experiment
#   bash scripts/run_experiments.sh [--dummy]    # Dummy mode (tiny runs)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
FAST_MODE=false
DUMMY_MODE=false
RUN_EXP=""
TOTAL_STEPS=200000
EVAL_EPS=100
EVAL_INTERVAL=20000
HAB_CONFIG=""

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --fast)   FAST_MODE=true ;;
        --dummy)  DUMMY_MODE=true ;;
        --exp)    shift; RUN_EXP="$1" ;;
        [0-9])    RUN_EXP="$arg" ;;
    esac
done

if [ "$DUMMY_MODE" = true ]; then
    echo "=== DUMMY MODE: tiny runs for pipeline testing ==="
    TOTAL_STEPS=200
    EVAL_EPS=5
    EVAL_INTERVAL=100
fi

# Common training flags
COMMON="--total_steps $TOTAL_STEPS --eval_episodes $EVAL_EPS --eval_interval $EVAL_INTERVAL"
if [ -n "$HAB_CONFIG" ]; then
    COMMON="$COMMON --habitat_config $HAB_CONFIG"
fi

DATA_DIR="datasets"
CKPT_DIR="checkpoints"
LOG_DIR="runs"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

# Helper: run one IQL experiment
run_iql() {
    local name="$1"
    local corr_type="$2"
    local dataset="$3"

    echo ""
    echo "================================================================"
    echo "  Running: $name"
    echo "  Model:   $corr_type | Dataset: $dataset"
    echo "================================================================"

    python3 -m training.train_iql \
        --dataset "$dataset" \
        --correlation_type "$corr_type" \
        --save_dir "$CKPT_DIR/$name" \
        --log_dir "$LOG_DIR/$name" \
        $COMMON

    # Evaluate best model
    if [ -f "$CKPT_DIR/$name/best_model.pt" ]; then
        python3 -m eval.evaluate \
            --checkpoint "$CKPT_DIR/$name/best_model.pt" \
            --num_episodes $EVAL_EPS \
            --output "$RESULTS_DIR/${name}.json" \
            --tag "$name"
    fi
}

# Helper: run one BC experiment
run_bc() {
    local name="$1"
    local corr_type="$2"
    local dataset="$3"

    echo ""
    echo "================================================================"
    echo "  Running BC: $name"
    echo "================================================================"

    python3 -m training.train_bc \
        --dataset "$dataset" \
        --correlation_type "$corr_type" \
        --save_dir "$CKPT_DIR/$name" \
        --log_dir "$LOG_DIR/$name" \
        $COMMON

    if [ -f "$CKPT_DIR/$name/best_model.pt" ]; then
        python3 -m eval.evaluate \
            --checkpoint "$CKPT_DIR/$name/best_model.pt" \
            --num_episodes $EVAL_EPS \
            --output "$RESULTS_DIR/${name}.json" \
            --tag "$name"
    fi
}

# ===================================================================
# Experiment 1: Core Question — Does correlation help offline?
# Fix: IQL, ideal (full expert) data. Vary: correlation type.
# ===================================================================
run_exp1() {
    echo ""
    echo "########################################################"
    echo "#  EXPERIMENT 1: Core Question                          #"
    echo "#  Does RSRNav correlation help in the offline setting?  #"
    echo "########################################################"

    local DS="$DATA_DIR/raw/expert/episodes.hdf5"
    if [ ! -f "$DS" ]; then
        DS="$DATA_DIR/raw/random/episodes.hdf5"
        echo "WARNING: Expert data not found, using random data at $DS"
    fi

    run_iql "exp1_semantic_ideal"   "semantic"         "$DS"
    run_iql "exp1_minimalist_ideal" "minimalist"       "$DS"
    run_iql "exp1_da_ideal"         "direction_aware"  "$DS"
}

# ===================================================================
# Experiment 2: Data Volume Sensitivity
# Fix: IQL, direction_aware + semantic. Vary: episode count.
# ===================================================================
run_exp2() {
    echo ""
    echo "########################################################"
    echo "#  EXPERIMENT 2: Data Volume Degradation Curve          #"
    echo "########################################################"

    for VOL in 9000 1000 250 50; do
        # Check if volume-reduced dataset exists
        if [ "$VOL" = "9000" ]; then
            DS="$DATA_DIR/raw/expert/episodes.hdf5"
        else
            DS="$DATA_DIR/vol-${VOL}/episodes.hdf5"
        fi

        if [ ! -f "$DS" ]; then
            echo "SKIP: $DS not found"
            continue
        fi

        run_iql "exp2_da_vol${VOL}"       "direction_aware" "$DS"
        run_iql "exp2_semantic_vol${VOL}"  "semantic"        "$DS"
    done
}

# ===================================================================
# Experiment 3: Demonstrator Quality Sensitivity
# Fix: IQL, direction_aware + semantic, 1k episodes. Vary: policy quality.
# ===================================================================
run_exp3() {
    echo ""
    echo "########################################################"
    echo "#  EXPERIMENT 3: Demonstrator Quality Sensitivity       #"
    echo "########################################################"

    for QUAL in expert noisy mixed random; do
        DS="$DATA_DIR/raw/${QUAL}/episodes.hdf5"
        if [ ! -f "$DS" ]; then
            echo "SKIP: $DS not found"
            continue
        fi

        run_iql "exp3_da_${QUAL}"       "direction_aware" "$DS"
        run_iql "exp3_semantic_${QUAL}"  "semantic"        "$DS"
    done
}

# ===================================================================
# Experiment 4: Combined Realistic Degradation
# Fix: IQL. Vary: model × degradation preset.
# ===================================================================
run_exp4() {
    echo ""
    echo "########################################################"
    echo "#  EXPERIMENT 4: Combined Realistic Degradation         #"
    echo "########################################################"

    for PRESET in mild moderate severe; do
        DS="$DATA_DIR/combined-${PRESET}/episodes.hdf5"
        if [ ! -f "$DS" ]; then
            echo "SKIP: $DS not found"
            continue
        fi

        run_iql "exp4_da_${PRESET}"         "direction_aware" "$DS"
        run_iql "exp4_minimalist_${PRESET}"  "minimalist"      "$DS"
        run_iql "exp4_semantic_${PRESET}"    "semantic"         "$DS"
    done
}

# ===================================================================
# Main execution
# ===================================================================

echo "============================================"
echo "  RSRNav Offline Experiments"
echo "============================================"
echo "  Mode: $([ "$FAST_MODE" = true ] && echo 'FAST (Exp 1+4)' || echo 'FULL')"
echo "  Steps/run: $TOTAL_STEPS"
echo "  Eval episodes: $EVAL_EPS"
echo ""

if [ -n "$RUN_EXP" ]; then
    echo "Running experiment $RUN_EXP only..."
    run_exp${RUN_EXP}
elif [ "$FAST_MODE" = true ]; then
    run_exp1
    run_exp4
else
    run_exp1
    run_exp2
    run_exp3
    run_exp4
fi

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "============================================"
echo ""
echo "Results:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  No results found."
echo ""
echo "Aggregate results with:"
echo "  python3 -m eval.aggregate_results"
