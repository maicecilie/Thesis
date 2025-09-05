#!/bin/bash
# init.sh - Python + CUDA + MPI environment for DTU HPC (Harvard-GF)

# ----------------- Configuration -----------------
VENV_NAME=venv
VENV_DIR=.
PYTHON_VERSION=3.11.9
CUDA_VERSION=11.8
MPI_MODULE=mpi/5.0.3-gcc-12.3.0-binutils-2.40   # Correct MPI module on DTU HPC

# ----------------- Load modules -----------------
module load python3/$PYTHON_VERSION
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "numpy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "scipy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "matplotlib/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "pandas/")
module load cuda/$CUDA_VERSION

# Load cuDNN
CUDNN_MOD=$(module avail -o modulepath -t cudnn 2>&1 | grep "cuda-${CUDA_VERSION}" | sort | tail -n1)
[[ ${CUDNN_MOD} ]] && module load ${CUDNN_MOD}

# Load MPI
module load $MPI_MODULE

# ----------------- Setup virtualenv -----------------
if [ ! -d "${VENV_DIR}/${VENV_NAME}" ]; then
    echo "INFO: Creating virtualenv..."
    virtualenv "${VENV_DIR}/${VENV_NAME}" --python=python3
fi
source "${VENV_DIR}/${VENV_NAME}/bin/activate"

# ----------------- Install required Python packages -----------------
# Only install if missing
pip install --upgrade pip

REQUIRED_PACKAGES=("torch" "scikit-learn" "mpi4py" "blobfile")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import ${pkg%%=*}" &> /dev/null; then
        echo "Installing $pkg..."
        if [ "$pkg" == "mpi4py" ]; then
            # Force compile from source so it links to the loaded MPI
            pip install --no-binary mpi4py mpi4py
        else
            pip install $pkg
        fi
    fi
done

# ----------------- Select least used GPU -----------------
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.gpu,index \
        --format=csv,noheader,nounits | sort -V | awk '{print $NF}' | head -n1)
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "Environment initialized."
