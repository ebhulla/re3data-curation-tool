# Negishi setup for the judge and fine-tuning jobs

Written by Claude on 2026-10-09 from the cluster notes in the project (account `surf`, partition `cpu`,
QOS `standby`). **Nothing here has been run yet.** Do the steps in order and stop at the first surprise.
Compute must never run on the login node: only via `sbatch` or `sinteractive`.

**Verified 2026-10-09:** a compute node reaches the internet (Groq returned HTTP 401, i.e. reachable); the setup in step 2 ran and 9 tests passed.

## 1. Get the code on Negishi (login node)
```bash
ssh <your-username>@negishi.rcac.purdue.edu
git clone https://github.com/ebhulla/re3data-curation-tool.git
cd re3data-curation-tool
git checkout fine-tuning-pipeline
```
If git asks for a password, use a GitHub personal access token (create one in GitHub settings; do not paste it into chat).

## 2. Find a Python and build the environment
```bash
module avail python anaconda 2>&1 | head -20
```
The system Python is 3.6.8 (too old). There is no plain `python` module; use anaconda:
```bash
module load anaconda/2024.10-py312
```
Then build the environment (do this inside `sinteractive`, not on the login node):
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy requests rapidfuzz scikit-learn sentence-transformers datasets accelerate pytest
python -m pytest tests -q
```
All 10 tests should pass. Then edit the `module load` / `source .venv/bin/activate` lines in
`slurm_judge.sh` and `slurm_finetune.sh` to match what you just did.

## 3. Download the embedding model once (login node, needs internet)
```bash
python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
```
Compute nodes may have no internet, so cache the model now.

## 4. Add the Groq key (you do this; never paste it into chat)
```bash
nano .env        # one line:  GROQ_API_KEY=...
chmod 600 .env
```

## 5. THE KEY TEST: can a compute node reach Groq?
```bash
sinteractive -A surf -p cpu -q standby -t 00:30:00 -N 1 -n 2
curl -sS -m 15 -o /dev/null -w "%{http_code}\n" https://api.groq.com/openai/v1/models
exit
```
A result like `401` or `200` means the node can reach Groq (401 is fine: no key was sent). A timeout or
connection error means compute nodes have no outbound internet. In that case tell Claude: the judge cannot run on
Negishi, and the fallback is to run it on the Codespace by hand each day. Fine-tuning does not need internet.

## 6. Start the judge job (only after the Codespace job is stopped and its labels are pushed)
```bash
sbatch slurm_judge.sh
squeue -u $USER
tail -f logs/judge_<jobid>.out
```
