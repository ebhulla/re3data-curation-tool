"""Does 'average of fuzzy and MiniLM' beat each alone across DIFFERENT random leakage-safe splits?

Written by Claude on 2026-10-14 -- FOR EKAM TO REVIEW. Read-only.

Re-splits the judged pairs 10 times (different hash salts; same no-name-in-two-splits rule as split_pairs.py),
picks each method's threshold on that split's train part, and scores on that split's test part.
Reports F1 and AP per method per split, then mean and spread across splits.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score

from baselines import best_threshold, prf, score_fuzzy

SALTS = range(10)
KEEP = {"duplicate", "hierarchy", "unrelated"}

groups = {}
for line in open("data/ror_matches.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    if rec["candidates"] and rec["candidates"][0]["chosen"]:
        groups[rec["name"]] = rec["candidates"][0]["ror_id"]

data = pd.concat([pd.read_json(p, lines=True) for p in ["data/judged_600.jsonl", "data/judged_mined.jsonl"]], ignore_index=True)
data = data[data.label.isin(KEEP)].reset_index(drop=True)
data["y"] = (data.label == "duplicate").astype(int)
print("judged usable pairs:", len(data), " duplicates:", data.y.sum())

# score every pair once with each method
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
names = sorted(set(data.name_a) | set(data.name_b))
vec = dict(zip(names, model.encode(names, normalize_embeddings=True, batch_size=64)))
data["fuzzy"] = [score_fuzzy(a, b) for a, b in zip(data.name_a, data.name_b)]
data["minilm"] = [float(vec[a] @ vec[b]) for a, b in zip(data.name_a, data.name_b)]
data["combo"] = (data.fuzzy + data.minilm) / 2


def split_of(key, salt):
    b = int(hashlib.md5(f"{salt}:{key}".encode("utf-8")).hexdigest(), 16) % 100
    return "train" if b < 80 else "val" if b < 90 else "test"


rows = []
for salt in SALTS:
    sa = [split_of(groups.get(n, n), salt) for n in data.name_a]
    sb = [split_of(groups.get(n, n), salt) for n in data.name_b]
    same = np.array([a == b for a, b in zip(sa, sb)])
    d = data[same].copy()
    d["split"] = np.array(sa)[same]
    tr, te = d[d.split == "train"], d[d.split == "test"]
    if te.y.sum() < 5:
        continue
    row = {"salt": salt, "test_pairs": len(te), "test_dups": int(te.y.sum())}
    for m in ("fuzzy", "minilm", "combo"):
        t = best_threshold(tr[m].values, tr.y.values)
        row[m + "_F1"] = prf(te[m].values, te.y.values, t)[2]
        row[m + "_AP"] = average_precision_score(te.y, te[m])
    rows.append(row)

res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
print("\nmean over splits (and std):")
for m in ("fuzzy", "minilm", "combo"):
    print(f"  {m:<7} F1 {res[m + '_F1'].mean():.2f} (+/-{res[m + '_F1'].std():.2f})   AP {res[m + '_AP'].mean():.2f} (+/-{res[m + '_AP'].std():.2f})")
wins = int((res.combo_F1 > res[["fuzzy_F1", "minilm_F1"]].max(axis=1)).sum())
winsap = int((res.combo_AP > res[["fuzzy_AP", "minilm_AP"]].max(axis=1)).sum())
print(f"\ncombo beats BOTH single methods on F1 in {wins}/{len(res)} splits, on AP in {winsap}/{len(res)} splits")
