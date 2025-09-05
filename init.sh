#!/bin/bash
# HPC Python init script with automatic package installation

# ---------------- Configuration ----------------
VENV_NAME=venv         # Name of your virtualenv
VENV_DIR=.             # Where to store your virtualenv
PYTHON_VERSION=3.11.9  # Python version
CUDA_VERSION=11.8      # CUDA version

# List of essential Python packages
REQUIRED_PACKAGES=(
    torch
    torchvision
    torchaudio
    scikit-learn
    fairlearn
    numpy
    scipy
    matplotlib
    pandas
)

# ---------------- Load Modules ----------------
module load python3/$PYTHON_VERSION
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "numpy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "scipy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "matplotlib/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "pandas/")
module load cuda/$CUDA_VERSION

# Load cuDNN for CUDA
CUDNN_MOD=$(module avail -o modulepath -t cudnn 2>&1 | grep "cuda-${CUDA_VERSION}" | sort | tail -n1)
if [ -z "$CUDNN_MOD" ]; then
    CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDNN_MOD=$(module avail -o modulepath -t cudnn 2>&1 | grep "cuda-${CUDA_VERSION_MAJOR}.X" | sort | tail -n1)
fi
if [[ ${CUDNN_MOD} ]]; then
    module load ${CUDNN_MOD}
fi

# ---------------- Virtual Environment ----------------
if [ ! -d "${VENV_DIR}/${VENV_NAME}" ]; then
    echo "INFO: Virtualenv not found. Creating..."
    virtualenv -p python3 "${VENV_DIR}/${VENV_NAME}"
fi
source "${VENV_DIR}/${VENV_NAME}/bin/activate"

# ---------------- Install Required Packages ----------------
echo "INFO: Installing missing Python packages..."
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import ${pkg%%=*}" &> /dev/null; then
        echo "Installing $pkg..."
        pip install --upgrade $pkg
    fi
done

# ---------------- GPU Selection ----------------
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.gpu,index \
        --format=csv,noheader,nounits | sort -V | awk '{print $NF}' | head -n1)
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "INFO: Initialization complete. Python environment ready."
