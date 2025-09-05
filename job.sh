#!/bin/sh
### General options 
### -- specify queue -- 
#BSUB -q gpuv100i        # GPU queue

#BSUB -gpu "num=1:mode=exclusive_process"  # request 1 GPU

### -- set the job Name -- 
#BSUB -J TrainGlaucoma

### -- ask for number of cores -- 
#BSUB -n 6 

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"

### -- set walltime limit hh:mm -- 3 days = 72:00
#BSUB -W 72:00

### -- send notification at start and end -- 
#BSUB -B 
#BSUB -N 

### -- Specify the output and error files. %J is job id -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# Prefer 32GB or NVLINK GPUs
# If none available, it falls back to standard GPUs
# (LSF will pick one that satisfies the resource)
#BSUB -R "select[gpu32gb || sxm2]"

# Load Python environment
source init.sh

# Go to the folder containing the Python training script
cd /work3/s185394/Thesis/Thesis-Repo/Harvard-GF-Repo

# Run the Python training script with your parameters
python3 train_glaucoma_fair.py \
    --data_dir /work3/s185394/Thesis/data/EyeFair \
    --result_dir ./results/crosssectional_rnflt_race/fullysup_efficientnet_rnflt_Taskcls_lr5e-5_bz6 \
    --model_type efficientnet \
    --image_size 200 \
    --loss_type bce \
    --lr 5e-5 --weight-decay 0. --momentum 0.1 \
    --batch-size 6 \
    --task cls \
    --epochs 10 \
    --modality_types rnflt \
    --perf_file efficientnet_rnflt_race.csv \
    --attribute_type race
