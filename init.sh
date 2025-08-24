#!/bin/bash
# Simple init script for Python on DTU HPC
# Patrick M. Jensen, patmjen@dtu.dk, 2022

# Configuration
# This is what you should change for your setup
VENV_NAME=venv         # Name of your virtualenv (default: venv)
VENV_DIR=.             # Where to store your virtualenv (default: current directory)
PYTHON_VERSION=3.11.9  # Python version (default: 3.11.9)
CUDA_VERSION=11.8      # CUDA version (default: 11.8)

# Load modules
module load python3/$PYTHON_VERSION
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "numpy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "scipy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "matplotlib/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "pandas/")
module load cuda/$CUDA_VERSION
CUDNN_MOD=$(module avail -o modulepath -t cudnn 2>&1 | grep "cuda-${CUDA_VERSION}" | sort | tail -n1)
if [ -z "$CUDNN_MOD" ]
then
    # Could find cuDNN for exact CUDA version, try to find major.X version
    CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDNN_MOD=$(module avail -o modulepath -t cudnn 2>&1 | grep "cuda-${CUDA_VERSION_MAJOR}.X" | sort | tail -n1)
fi
if [[ ${CUDNN_MOD} ]]
then
    module load ${CUDNN_MOD}
fi

# Create virtualenv if needed and activate it
if [ ! -d "${VENV_DIR}/${VENV_NAME}" ]
then
    echo INFO: Did not find virtualenv. Creating...
    virtualenv "${VENV_DIR}/${VENV_NAME}"
fi
source "${VENV_DIR}/${VENV_NAME}/bin/activate"

# Select least used GPU if any are available
if command -v nvidia-smi &> /dev/null
then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.gpu,index --format=csv,noheader,nounits | sort -V | awk '{print $NF}' | head -n1)
    echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
fi

