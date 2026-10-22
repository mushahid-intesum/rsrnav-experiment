"""
Central configuration for all offline RL experiments.

All experiment constants live here. Training scripts import from this file.
"""
import os

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoint", "checkpoint.pth")

# Gibson scene data — update this to your actual Gibson data directory
GIBSON_SCENE_DIR = os.path.join(PROJECT_ROOT, "data", "scene_datasets", "gibson")

# Episode datasets — update to where train/val JSON.gz files live
EPISODE_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "datasets", "imagenav", "gibson", "v2")

# Habitat config
HABITAT_CONFIG = os.path.join(PROJECT_ROOT, "exp_config", "ddppo_imagenav_gibson.yaml")
HABITAT_OVERRIDES = [
    os.path.join(PROJECT_ROOT, "src", "config", "policy.yaml"),
    os.path.join(PROJECT_ROOT, "src", "config", "reward.yaml"),
    os.path.join(PROJECT_ROOT, "src", "config", "dataset.yaml"),
    os.path.join(PROJECT_ROOT, "src", "config", "sensors.yaml"),
    os.path.join(PROJECT_ROOT, "src", "config", "RSRNav.yaml"),
]

# Output directories
OFFLINE_DATA_DIR = os.path.join(PROJECT_ROOT, "offline_data")
RAW_DATA_DIR = os.path.join(OFFLINE_DATA_DIR, "raw")
DEGRADED_DATA_DIR = os.path.join(OFFLINE_DATA_DIR, "degraded")
RESULTS_DIR = os.path.join(OFFLINE_DATA_DIR, "results")
LOG_DIR = os.path.join(PROJECT_ROOT, "offline_runs")

# ==============================================================================
# DATA COLLECTION
# ==============================================================================
COLLECT_NUM_EPISODES = 1000
COLLECT_MAX_STEPS = 500
COLLECT_POLICIES = ["expert", "noisy_expert", "mixed", "random"]

# ==============================================================================
# TRAINING — IQL
# ==============================================================================
IQL_LR = 3e-4
IQL_BATCH_SIZE = 64
IQL_SEQ_LEN = 64
IQL_STRIDE = 32
IQL_TOTAL_STEPS = 200_000
IQL_BETA = 3.0           # AWR temperature
IQL_TAU = 0.7            # expectile for value function
IQL_GAMMA = 0.99
IQL_TARGET_UPDATE_RATE = 0.005
IQL_SAVE_INTERVAL = 10_000
IQL_EVAL_INTERVAL = 20_000
IQL_LOG_INTERVAL = 100

# ==============================================================================
# TRAINING — BC
# ==============================================================================
BC_LR = 3e-4
BC_BATCH_SIZE = 64
BC_SEQ_LEN = 64
BC_STRIDE = 32
BC_TOTAL_STEPS = 100_000
BC_SAVE_INTERVAL = 10_000
BC_EVAL_INTERVAL = 20_000
BC_LOG_INTERVAL = 100

# ==============================================================================
# EVALUATION
# ==============================================================================
EVAL_NUM_EPISODES = 100
EVAL_MAX_STEPS = 500

# ==============================================================================
# WANDB
# ==============================================================================
WANDB_PROJECT = "rsrnav-offline"
WANDB_ENTITY = None  # set to your wandb username/team
