"""Baselines for 'are these two institution names the SAME institution?' (duplicate vs everything else).

Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.

Task: positive = judge label 'duplicate'; negative = 'hierarchy' or 'unrelated'.
Hierarchy pairs are the hard negatives: one name contains the other but they are NOT the same entity.

Methods (all produce a score; higher = more likely duplicate):
  exact_norm   normalized strings are equal (lowercase, punctuation stripped)
  substring    one normalized name contains the other  <- the naive rule from the Findings report
  fuzzy        rapidfuzz token_sort_ratio
  minilm       cosine similarity of off-the-shelf all-MiniLM-L6-v2 vectors

Threshold rule: pick the threshold with best F1 on TRAIN, apply it unchanged to TEST.
Also reports average precision (threshold-free) and the mean score per label, to show separation.

Usage: python baselines.py [model_name_or_path]   (default: off-the-shelf MiniLM)
"""
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, precision_recall_curve

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", s.lower())).strip()


def score_exact(a, b):
    return float(norm(a) == norm(b))


def score_substring(a, b):
    na, nb = norm(a), norm(b)
    return float(na in nb or nb in na)


def score_fuzzy(a, b):
    return fuzz.token_sort_ratio(norm(a), norm(b)) / 100


def minilm_scores(df, model_name):
    model = SentenceTransformer(model_name)
    names = sorted(set(df.name_a) | set(df.name_b))
    vec = dict(zip(names, model.encode(names, normalize_embeddings=True, batch_size=64)))
    return np.array([float(vec[a] @ vec[b]) for a, b in zip(df.name_a, df.name_b)])


def best_threshold(scores, y):
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.argmax(f1[:-1])) if len(thr) else 0
    return thr[i] if len(thr) else 0.5


def prf(scores, y, t):
    pred = scores >= t
    tp = int((pred & (y == 1)).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int((y == 1).sum()), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    train = pd.read_csv("data/pairs_train.csv")
    test = pd.read_csv("data/pairs_test.csv")
    print(f"train: {len(train)} pairs ({train.is_duplicate.sum()} duplicates)   test: {len(test)} pairs ({test.is_duplicate.sum()} duplicates)\n")

    methods = {
        "exact_norm": lambda d: np.array([score_exact(a, b) for a, b in zip(d.name_a, d.name_b)]),
        "substring": lambda d: np.array([score_substring(a, b) for a, b in zip(d.name_a, d.name_b)]),
        "fuzzy": lambda d: np.array([score_fuzzy(a, b) for a, b in zip(d.name_a, d.name_b)]),
        "minilm": lambda d: minilm_scores(d, MODEL),
    }
    rows = []
    for name, fn in methods.items():
        s_tr, s_te = fn(train), fn(test)
        t = best_threshold(s_tr, train.is_duplicate.values)
        p, r, f = prf(s_te, test.is_duplicate.values, t)
        ap = average_precision_score(test.is_duplicate, s_te) if test.is_duplicate.sum() else float("nan")
        means = {lab: s_te[(test.label == lab).values].mean() for lab in ["duplicate", "hierarchy", "unrelated"]}
        rows.append({"method": name, "threshold": round(float(t), 3), "precision": round(p, 2), "recall": round(r, 2),
                     "F1": round(f, 2), "AP": round(float(ap), 2),
                     "mean_dup": round(means["duplicate"], 2), "mean_hier": round(means["hierarchy"], 2),
                     "mean_unrel": round(means["unrelated"], 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nmean_* = average score of test pairs with that judge label; a good method has mean_dup >> mean_hier.")


if __name__ == "__main__":
    main()
