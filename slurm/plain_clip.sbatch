#!/bin/bash
# GPU job: plain CLIP baseline on all clips and all text sets, then evaluation.
# Under a minute of GPU time, so the 15-minute test queue is enough.
# Submit from the repo root on roihu-gpu.csc.fi:
#   source roihu/env.sh && sbatch slurm/plain_clip.sbatch
#SBATCH --job-name=dfer_plain_clip
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:15:00
#SBATCH --output=logs/plain_clip_%j.out

source roihu/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

OUT_DIR="${DFER_DATA_DIR}/results/plain_clip"
srun python src/plain_clip.py --out_dir "${OUT_DIR}" --workers 16 --batch_size 64
srun python src/evaluate.py "${OUT_DIR}"/*.csv --out_dir "${OUT_DIR}/evaluation"
srun python src/significance.py --results_dir "${OUT_DIR}"
