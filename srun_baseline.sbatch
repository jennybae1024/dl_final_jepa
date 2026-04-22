#!/bin/bash
#SBATCH --job-name=dlfl_baseline
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --open-mode=append # Append to the output file. If you don't use this, the output will be overwritten.
#SBATCH --output=./%j_%x.out # %j is the job id, %x is the job name.
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL # Export all the environment variables to the job.
#SBATCH --time=10:00:00 
#SBATCH --gres=gpu:2
#SBATCH --requeue # Requeue the job if it fails.

singularity exec --bind /scratch --nv --overlay  /scratch/$USER/overlay-25GB-500K.ext3:ro /scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -lc "
    source ~/.bashrc
    conda activate dlfl 
    cd /scratch/$USER/dl_final/physical-representation-learning
    bash scripts/active_matter/run_train_jepa.sh train.num_epochs=20 train.lr=5e-4 train.save_every=1
  "

