#!/bin/bash
# CPU job: 16 parallel shards. Submit from the repo root on roihu-cpu.csc.fi:
#   source roihu/env.sh && sbatch slurm/extract_frames.sbatch
#SBATCH --job-name=dfer_extract
#SBATCH --partition=small
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=logs/extract_%A_%a.out

source roihu/env.sh
srun python src/extract_frames.py \
    --shard_index "${SLURM_ARRAY_TASK_ID}" --num_shards 16