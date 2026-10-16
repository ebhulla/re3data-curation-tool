#!/bin/bash
# Run the LLM judge on Negishi and keep going day after day until all mined pairs are labeled.
#
# Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.
# Environment verified on a compute node; the job itself has not been submitted yet. See docs/negishi_setup.md.
#
# Each run FIRST queues its successor (~23 hours later, up to MAX_RESUBMITS times), THEN runs judge.py,
# which resumes from data/judged_mined.jsonl until Groq's daily token cap stops it. Queuing first matters:
# if the cap makes judge.py wait past the time limit, Slurm kills this job, but the next one is already queued.
# Only ONE machine may write data/judged_mined.jsonl at a time (stop the Codespace job first).
#SBATCH --account=surf
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=judge_mined
#SBATCH --output=logs/judge_%j.out

set -uo pipefail
mkdir -p logs
TOTAL=2800
MAX_RESUBMITS=5
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

# --- environment (confirmed on Negishi 2026-10-09) ---
module load anaconda/2024.10-py312
source .venv/bin/activate

DONE=$(wc -l < data/judged_mined.jsonl)
if [ "${DONE}" -ge "${TOTAL}" ]; then
    echo "all ${TOTAL} pairs already labeled; nothing to do"
    exit 0
fi

if [ "${RESUBMIT_COUNT}" -lt "${MAX_RESUBMITS}" ]; then
    NEXT=$(( RESUBMIT_COUNT + 1 ))
    sbatch --begin=now+23hours --export=ALL,RESUBMIT_COUNT=${NEXT} slurm_judge.sh
    echo "queued successor ${NEXT}/${MAX_RESUBMITS} to start in ~23 hours"
fi

python judge.py --input data/mined_pairs.csv --out data/judged_mined.jsonl --limit ${TOTAL} || echo "judge.py exited early (expected when the daily cap is hit)"
echo "labeled at end of run: $(wc -l < data/judged_mined.jsonl) / ${TOTAL}"
