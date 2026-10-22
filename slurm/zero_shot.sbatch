#!/bin/bash
# GPU job: every prompt set on every checkpoint, then the summary table.
# Submit from the repo root on roihu-gpu.csc.fi:
#   source roihu/env.sh && sbatch slurm/zero_shot.sbatch
# Quick test first (200 clips, one prompt set, 15-minute test queue):
#   PROMPT_SETS=class_descriptor LIMIT=200 sbatch --partition=gputest --time=00:15:00 slurm/zero_shot.sbatch
#SBATCH --job-name=dfer_zero_shot
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=logs/zero_shot_%j.out

source roihu/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

CHECKPOINTS=${CHECKPOINTS:-"${DFER_DATA_DIR}/checkpoints/dfew_fold1.pth"}
PROMPT_SETS=${PROMPT_SETS:-"class_descriptor class_names class_names_with_context facs other_llm shuffled"}
LIMIT_ARG=${LIMIT:+--limit ${LIMIT}}

for checkpoint in ${CHECKPOINTS}; do
    for prompt_set in ${PROMPT_SETS}; do
        srun python src/zero_shot.py --checkpoint "${checkpoint}" --prompt_set "${prompt_set}" \
            --workers 16 --batch_size 64 ${LIMIT_ARG}
    done
done
srun python src/evaluate.py