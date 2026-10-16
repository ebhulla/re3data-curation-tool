"""Summarize the fine-tuning sweep from the Slurm logs (run on Negishi, in the repo folder).

Written by Claude on 2026-10-14 -- FOR EKAM TO REVIEW.

Reads logs/ft_*.out. Each finished task prints 'saved to models/ft_lr<LR>_s<SEED>' and then a result row for the
fine-tuned model (the row whose method is 'minilm'). Prints one line per run, then mean and range per learning rate,
next to the off-the-shelf MiniLM and fuzzy baselines from the same logs.
"""
import glob
import re
import statistics as st

runs = []
baseline = {}
for path in sorted(glob.glob("logs/ft_*.out")):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    tag = next((re.search(r"ft_lr(\S+)_s(\d+)", l) for l in lines if l.startswith("saved to")), None)
    rows = {}
    for l in lines:
        m = re.match(r"\s*(exact_norm|substring|fuzzy|minilm)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", l)
        if m:
            rows[m.group(1)] = dict(precision=float(m.group(3)), recall=float(m.group(4)), f1=float(m.group(5)), ap=float(m.group(6)))
    if tag and "minilm" in rows:
        runs.append((tag.group(1), int(tag.group(2)), rows["minilm"]))
        baseline.update({k: v for k, v in rows.items() if k in ("fuzzy", "exact_norm", "substring")})

print(f"{len(runs)} finished runs\n")
print("lr      seed  precision  recall    F1     AP")
for lr, seed, r in sorted(runs):
    print(f"{lr:<7} {seed:<5} {r['precision']:<10} {r['recall']:<8} {r['f1']:<6} {r['ap']}")

print("\nper learning rate (mean, min-max over seeds):")
for lr in sorted({lr for lr, _, _ in runs}):
    f1 = [r["f1"] for l, _, r in runs if l == lr]
    ap = [r["ap"] for l, _, r in runs if l == lr]
    print(f"  lr {lr}: F1 {st.mean(f1):.2f} ({min(f1):.2f}-{max(f1):.2f})   AP {st.mean(ap):.2f} ({min(ap):.2f}-{max(ap):.2f})   n={len(f1)}")

all_f1 = [r["f1"] for _, _, r in runs]
all_ap = [r["ap"] for _, _, r in runs]
if all_f1:
    print(f"\nall runs: F1 {st.mean(all_f1):.2f} (range {min(all_f1):.2f}-{max(all_f1):.2f})   AP {st.mean(all_ap):.2f} (range {min(all_ap):.2f}-{max(all_ap):.2f})")
for name, r in baseline.items():
    print(f"baseline {name}: F1 {r['f1']}   AP {r['ap']}")
print("off-the-shelf MiniLM (not in these logs; current split, 111 test pairs): F1 0.81, AP 0.90 -- update if the split changes")
