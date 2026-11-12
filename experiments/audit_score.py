"""Prepare and score the audit sheets using the AI judge's labels. Run from the repo root.

  1.  python experiments/audit_score.py --prepare
        -> writes data/audit_pairs.csv (every pair from audit/A, B, C, de-duplicated)
  2.  python pipeline/judge.py --input data/audit_pairs.csv --out data/judged_audit.jsonl      (needs the Groq key; ~300 pairs)
  3.  python experiments/audit_score.py
        -> prints merge precision (A), tool vs baselines (B), parent accuracy (C), with 95% Wilson intervals.

Judge label -> audit meaning:  duplicate = same, hierarchy = part, unrelated = different.
Pairs the judge could not settle (the two name orders disagree, or 'uncertain') are excluded and counted, and a worst-case
number (undecided counted as wrong) is shown next to every headline figure.
Developed with AI assistance (Claude); see docs/AUTHORSHIP.md.
"""
import json
import sys

import numpy as np
import pandas as pd

A, B, C = "audit/A_merges.csv", "audit/B_candidates.csv", "audit/C_parents.csv"
PAIRS, JUDGED = "data/audit_pairs.csv", "data/judged_audit.jsonl"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt(k, n, total=None):
    lo, hi = wilson(k, n)
    extra = ""
    if total is not None and total > n:
        extra = f"   (worst case, {total - n} undecided counted wrong: {k / total:.0%})"
    return f"{k}/{n} = {k / n:.0%}   95% interval {lo:.0%} to {hi:.0%}{extra}" if n else "no decided pairs"


def prepare():
    a, b, c = pd.read_csv(A), pd.read_csv(B), pd.read_csv(C)
    pairs = list(zip(a.name_a, a.name_b)) + list(zip(b.name, b.candidate)) + list(zip(c.unit, c.parent))
    seen, rows = set(), []
    for x, y in pairs:
        if (x, y) not in seen and (y, x) not in seen:
            seen.add((x, y))
            rows.append({"name_a": x, "name_b": y})
    pd.DataFrame(rows).to_csv(PAIRS, index=False, encoding="utf-8")
    print(f"wrote {PAIRS}: {len(rows)} unique pairs (of {len(pairs)} sheet rows)")


def load_labels():
    lab = {}
    for line in open(JUDGED, encoding="utf-8"):
        r = json.loads(line)
        lab[(r["name_a"], r["name_b"])] = r["label"]
    return lab


def get(lab, x, y):
    return lab.get((x, y), lab.get((y, x), "missing"))


def score():
    lab = load_labels()
    a, b, c = pd.read_csv(A), pd.read_csv(B), pd.read_csv(C)

    # ---- A: merge precision
    a["judge"] = [get(lab, x, y) for x, y in zip(a.name_a, a.name_b)]
    decided = a[a.judge.isin(["duplicate", "hierarchy", "unrelated"])]
    k = int((decided.judge == "duplicate").sum())
    print("== A. Of the pairs the tool MERGED, how many are really the same institution?")
    print("   judge says:", a.judge.value_counts().to_dict())
    print("   merge precision:", fmt(k, len(decided), len(a)))
    print("   (a merge is correct only when the judge says duplicate; 'hierarchy' means a unit was merged into its parent)")

    # ---- B: tool vs baselines on the same candidate rows
    b["judge"] = [get(lab, x, y) for x, y in zip(b.name, b.candidate)]
    d = b[b.judge.isin(["duplicate", "hierarchy", "unrelated"])].copy()
    d["truth"] = d.judge == "duplicate"
    print("\n== B. Tool vs simple methods on the same lookalike pairs "
          f"({d.name_id.nunique()} names, {len(d)} decided pairs, {int(d.truth.sum())} truly the same)")
    print(f"   {'method':<26}{'merges':>7}{'correct':>9}{'precision':>11}{'recall':>9}")
    methods = {"do nothing": pd.Series(False, index=d.index), "exact match after cleaning": d.exact_after_cleaning,
               "fuzzy matching": d.fuzzy_ge_threshold, "OUR TOOL": d.tool_same_cluster}
    for name, pred in methods.items():
        tp = int((pred & d.truth).sum())
        n_pred = int(pred.sum())
        prec = f"{tp / n_pred:.0%}" if n_pred else "n/a"
        rec = f"{tp / d.truth.sum():.0%}" if d.truth.sum() else "n/a"
        print(f"   {name:<26}{n_pred:>7}{tp:>9}{prec:>11}{rec:>9}")
    print("   Recall here is over names that have lookalikes in the list, not over all names.")

    # ---- C: parent links
    c["judge"] = [get(lab, x, y) for x, y in zip(c.unit, c.parent)]
    dc = c[c.judge.isin(["duplicate", "hierarchy", "unrelated"])]
    k = int((dc.judge == "hierarchy").sum())
    print("\n== C. Of the parent links, how many point to the institution that really contains the unit?")
    print("   judge says:", c.judge.value_counts().to_dict())
    print("   parent accuracy:", fmt(k, len(dc), len(c)))


if __name__ == "__main__":
    prepare() if "--prepare" in sys.argv else score()
