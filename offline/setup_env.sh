#!/usr/bin/env bash
#
# Setup script for RSRNav offline RL experiments.
# Creates a conda environment matching the official RSRNav dependencies.
#
# Usage:
#   chmod +x offline/setup_env.sh
#   ./offline/setup_env.sh
#
set -e

ENV_NAME="rsrnav"
PYTHON_VERSION="3.9"

echo "=== RSRNav Offline RL Environment Setup ==="
echo ""

# ─── Step 1: Create conda env ───
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

echo ""
echo "Activating ${ENV_NAME}..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# ─── Step 2: Install habitat-sim ───
echo ""
echo "=== Installing habitat-sim ==="
# habitat-sim 0.2.2 headless (no display needed for data collection)
conda install habitat-sim=0.2.2 withbullet headless -c conda-forge -c aihabitat -y || {
    echo "WARNING: habitat-sim 0.2.2 install failed."
    echo "Try: conda install habitat-sim headless -c conda-forge -c aihabitat -y"
    echo "Or build from source: https://github.com/facebookresearch/habitat-sim"
}

# ─── Step 3: Install PyTorch ───
echo ""
echo "=== Installing PyTorch ==="
# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "Detected CUDA: ${CUDA_VERSION}"

    # Install torch with CUDA support
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "No CUDA detected. Installing CPU torch."
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html
fi

# ─── Step 4: Install habitat-lab ───
echo ""
echo "=== Installing habitat-lab ==="
pip install habitat-lab || {
    echo "WARNING: habitat-lab pip install failed."
    echo "You may need to install from source:"
    echo "  git clone https://github.com/facebookresearch/habitat-lab.git"
    echo "  cd habitat-lab && pip install -e habitat-lab"
}

# ─── Step 5: Install remaining dependencies ───
echo ""
echo "=== Installing remaining dependencies ==="
pip install \
    numpy\<2.0 \
    h5py \
    wandb \
    tensorboard \
    scipy \
    opencv-python\<4.10 \
    pyyaml \
    tqdm \
    einops \
    transformers \
    clip \
    pytorch-grad-cam

# ─── Step 6: Install this project in dev mode ───
echo ""
echo "=== Installing RSRNav in dev mode ==="
cd "$(dirname "$0")/.."
pip install -e .  2>/dev/null || echo "No setup.py found, that's OK."

# ─── Step 7: Verify ───
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import habitat_sim; print(f'habitat-sim: {habitat_sim.__version__}')" 2>/dev/null || echo "habitat-sim: not loaded"
python -c "import habitat; print(f'habitat-lab: OK')" 2>/dev/null || echo "habitat-lab: not loaded"
python -c "import h5py; print(f'h5py: {h5py.__version__}')"
python -c "import wandb; print(f'wandb: {wandb.__version__}')"
echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. Verify checkpoint loads: python offline/inspect_checkpoint.py"
echo "  2. Ensure Gibson data is at: data/scene_datasets/gibson/"
echo "  3. Ensure episode data is at: data/datasets/imagenav/gibson/v2/"
