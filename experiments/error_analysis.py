"""Error analysis on the 82-pair test set: where do fuzzy, MiniLM and fine-tuned MiniLM go wrong?

Read-only: it changes no data.

Usage: python experiments/error_analysis.py [path_to_finetuned_model]
Thresholds are the best-F1 thresholds picked on the TRAIN pairs (same rule as experiments/baselines.py).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from baselines import best_threshold, minilm_scores, prf, score_fuzzy

FT = sys.argv[1] if len(sys.argv) > 1 else None
train = pd.read_csv("data/pairs_train.csv")
test = pd.read_csv("data/pairs_test.csv")
ytr, yte = train.is_duplicate.values, test.is_duplicate.values

methods = {"fuzzy": lambda d: np.array([score_fuzzy(a, b) for a, b in zip(d.name_a, d.name_b)]),
           "minilm": lambda d: minilm_scores(d, "sentence-transformers/all-MiniLM-L6-v2")}
if FT:
    methods["finetuned"] = lambda d: minilm_scores(d, FT)

scores, thr, pred = {}, {}, {}
for name, fn in methods.items():
    s_tr, s_te = fn(train), fn(test)
    thr[name] = best_threshold(s_tr, ytr)
    scores[name] = s_te
    pred[name] = s_te >= thr[name]

# simple combination: average of fuzzy and off-the-shelf MiniLM, threshold picked on train
combo_tr = (methods["fuzzy"](train) + methods["minilm"](train)) / 2
thr["combo"] = best_threshold(combo_tr, ytr)
scores["combo"] = (scores["fuzzy"] + scores["minilm"]) / 2
pred["combo"] = scores["combo"] >= thr["combo"]

print(f"test pairs: {len(test)}  duplicates: {yte.sum()}\n")
print("method      threshold  false alarms (FP)  misses (FN)   F1    AP")
for name in scores:
    p, r, f = prf(scores[name], yte, thr[name])
    fp = int((pred[name] & (yte == 0)).sum())
    fn_ = int((~pred[name] & (yte == 1)).sum())
    print(f"{name:<11} {thr[name]:<10.3f} {fp:<18} {fn_:<13} {f:.2f}  {average_precision_score(yte, scores[name]):.2f}")

# which duplicates does each method miss? are the misses the same pairs?
dup = test[yte == 1].copy()
for name in scores:
    dup[name] = scores[name][yte == 1]
    dup[name + "_hit"] = pred[name][yte == 1]
print("\nduplicates MISSED by fuzzy or MiniLM (score shown; threshold in brackets):")
print(f"  fuzzy [{thr['fuzzy']:.2f}]  minilm [{thr['minilm']:.2f}]")
missed = dup[~dup.fuzzy_hit | ~dup.minilm_hit]
for _, r in missed.iterrows():
    print(f"  fuzzy {r.fuzzy:.2f} {'ok ' if r.fuzzy_hit else 'MISS'}  minilm {r.minilm:.2f} {'ok ' if r.minilm_hit else 'MISS'} | {r.name_a[:55]} | {r.name_b[:55]}")
both = int((~dup.fuzzy_hit & ~dup.minilm_hit).sum())
print(f"  missed by both: {both}   only fuzzy missed: {int((~dup.fuzzy_hit & dup.minilm_hit).sum())}   only MiniLM missed: {int((dup.fuzzy_hit & ~dup.minilm_hit).sum())}")

print("\nFALSE ALARMS (judge says NOT a duplicate, method says duplicate):")
nd = test[yte == 0].copy()
for name in ("fuzzy", "minilm"):
    nd[name] = scores[name][yte == 0]
    nd[name + "_flag"] = pred[name][yte == 0]
for _, r in nd[nd.fuzzy_flag | nd.minilm_flag].iterrows():
    print(f"  [{r.label}] fuzzy {r.fuzzy:.2f}{'*' if r.fuzzy_flag else ' '} minilm {r.minilm:.2f}{'*' if r.minilm_flag else ' '} | {r.name_a[:50]} | {r.name_b[:50]}")
print("  (* = this method flagged it)")
