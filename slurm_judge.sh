#!/bin/bash
# Run the LLM judge on Negishi and keep going day after day until all mined pairs are labeled.
#
# Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.
# NOT YET RUN on Negishi. Follow docs/negishi_setup.md first (environment, .env key, internet test).
#
# Each run: judge.py resumes from data/judged_mined.jsonl until Groq's daily token cap stops it,
# then this script resubmits itself to start ~23 hours later, up to MAX_RESUBMITS times.
# Only ONE machine may write data/judged_mined.jsonl at a time (stop the Codespace job first).
#SBATCH --account=surf
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=judge_mined
#SBATCH --output=logs/judge_%j.out

set -uo pipefail
mkdir -p logs
TOTAL=2800
MAX_RESUBMITS=5
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

# --- environment (EDIT to match docs/negishi_setup.md) ---
# module load anaconda
# source .venv/bin/activate

python judge.py --input data/mined_pairs.csv --out data/judged_mined.jsonl --limit ${TOTAL} || echo "judge.py exited early (expected when the daily cap is hit)"

DONE=$(wc -l < data/judged_mined.jsonl)
echo "labeled so far: ${DONE} / ${TOTAL}"

if [ "${DONE}" -lt "${TOTAL}" ] && [ "${RESUBMIT_COUNT}" -lt "${MAX_RESUBMITS}" ]; then
    NEXT=$(( RESUBMIT_COUNT + 1 ))
    sbatch --begin=now+23hours --export=ALL,RESUBMIT_COUNT=${NEXT} slurm_judge.sh
    echo "resubmitted (${NEXT}/${MAX_RESUBMITS}) to start in ~23 hours"
fi
