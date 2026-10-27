"""Does the name-only combined model beat fuzzy matching alone across 10 different leakage-safe splits?

Plumbing around pair_model.py's features (copied, name-only: no ROR features).
For each split: train 3-class gradient boosting on that split's train part, score on its test part.
Reports duplicate-vs-rest F1 and AP, and hierarchy F1, for the model vs fuzzy alone (threshold picked on train).
"""
import hashlib
import json

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

from baselines import best_threshold, norm, prf

KEEP = {"duplicate", "hierarchy", "unrelated"}
groups = {}
for line in open("data/ror_slim.jsonl", encoding="utf-8"):
    r = json.loads(line)
    groups[r["name"]] = r["ror_id"]          # only used to keep a ROR cluster inside one split, never as a feature

data = pd.concat([pd.read_json(p, lines=True) for p in ["data/judged_600.jsonl", "data/judged_mined.jsonl"]], ignore_index=True)
data = data[data.label.isin(KEEP)].reset_index(drop=True)
names = sorted(set(data.name_a) | set(data.name_b))
vec = dict(zip(names, SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(names, normalize_embeddings=True, batch_size=64)))


def features(a, b):
    na, nb = norm(a), norm(b)
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    return {
        "fuzzy_sort": fuzz.token_sort_ratio(na, nb) / 100, "fuzzy_set": fuzz.token_set_ratio(na, nb) / 100,
        "fuzzy_partial": fuzz.partial_ratio(na, nb) / 100, "minilm_cos": float(vec[a] @ vec[b]),
        "contains": float(short in long_), "len_ratio": len(short) / max(len(long_), 1),
        "extra_words": abs(len(na.split()) - len(nb.split())), "has_comma_diff": float(("," in a) != ("," in b)),
    }


X = pd.DataFrame([features(a, b) for a, b in zip(data.name_a, data.name_b)])
y_dup = (data.label == "duplicate").astype(int).values


def split_of(key, salt):
    b = int(hashlib.md5(f"{salt}:{key}".encode("utf-8")).hexdigest(), 16) % 100
    return "train" if b < 80 else "test" if b >= 90 else "val"


rows = []
for salt in range(10):
    sa = np.array([split_of(groups.get(n, n), salt) for n in data.name_a])
    sb = np.array([split_of(groups.get(n, n), salt) for n in data.name_b])
    same = sa == sb
    tr, te = same & (sa == "train"), same & (sa == "test")
    if (y_dup[te] == 1).sum() < 8:
        continue
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X[tr], data.label[tr])
    proba = model.predict_proba(X[te])
    p_dup = proba[:, list(model.classes_).index("duplicate")]
    pred = model.predict(X[te])
    thr_f = best_threshold(X.fuzzy_sort[tr].values, y_dup[tr])
    rows.append({
        "salt": salt, "test_dups": int(y_dup[te].sum()),
        "model_F1": prf(p_dup, y_dup[te], 0.5)[2], "model_AP": average_precision_score(y_dup[te], p_dup),
        "fuzzy_F1": prf(X.fuzzy_sort[te].values, y_dup[te], thr_f)[2], "fuzzy_AP": average_precision_score(y_dup[te], X.fuzzy_sort[te]),
        "model_hier_F1": f1_score(data.label[te] == "hierarchy", pred == "hierarchy"),
    })
res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
print()
for m in ("model", "fuzzy"):
    print(f"{m:<6} duplicate-vs-rest F1 {res[m + '_F1'].mean():.2f} (+/-{res[m + '_F1'].std():.2f})   AP {res[m + '_AP'].mean():.2f} (+/-{res[m + '_AP'].std():.2f})")
print(f"model hierarchy F1 {res.model_hier_F1.mean():.2f} (+/-{res.model_hier_F1.std():.2f})")
print(f"model beats fuzzy on F1 in {(res.model_F1 > res.fuzzy_F1).sum()}/{len(res)} splits, on AP in {(res.model_AP > res.fuzzy_AP).sum()}/{len(res)} splits")
