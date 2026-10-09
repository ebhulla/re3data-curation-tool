#!/bin/bash
# Fine-tuning sweep on Negishi: 3 learning rates x 2 seeds = 6 array tasks, CPU only.
#
# Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.
# NOT YET RUN on Negishi. First submit ONE task as a test:   sbatch --array=0 slurm_finetune.sh
# Then the full sweep:                                       sbatch --array=0-5 slurm_finetune.sh
#
# Settings below come from the confirmed-working test job notes: account surf, partition cpu,
# QOS standby (free, idle nodes, under 4 hours). Edit the 'module load' / venv lines to match the
# environment you actually build on Negishi (that step is the unverified part).
#SBATCH --account=surf
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ft_institutions
#SBATCH --output=logs/ft_%A_%a.out

set -euo pipefail
mkdir -p logs models

LRS=(1e-5 2e-5 5e-5)
SEEDS=(1 2)
LR=${LRS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
OUT=models/ft_lr${LR}_s${SEED}

# --- environment (confirmed on Negishi 2026-10-09) ---
module load anaconda/2024.10-py312
source .venv/bin/activate

echo "task ${SLURM_ARRAY_TASK_ID}: lr=${LR} seed=${SEED} -> ${OUT}"
python finetune.py --lr "${LR}" --epochs 3 --seed "${SEED}" --out "${OUT}"
python baselines.py "${OUT}"
