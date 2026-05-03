#!/bin/bash
# Edit this file to match your computing environment, then source it from the
# other scripts (it is sourced automatically by all scripts in this directory).

# Load Python module if your cluster uses environment modules, e.g.:
# module load python/3.11.7

source ~/.bashrc

export HF_HOME=/scratch/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/$USER/huggingface
export THE_WELL_DATA_DIR=/scratch/$USER/huggingface

conda activate /scratch/$USER/envs/dlfl

cd /scratch/$USER/dl_final_jepa
