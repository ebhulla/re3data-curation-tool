"""Worked example: predict duplicate / hierarchy / unrelated for a pair of institution names by combining several signals.

A worked example to read, run and modify (see the missions at the bottom).

The idea: no single score separates the three classes (fuzzy misses names with extra words, MiniLM thinks a sub-unit
and its parent are near-duplicates). A model can learn how to weigh several signals together.

    python pair_model.py            # all features
    python pair_model.py --no-ror   # drop the ROR features (see mission 3 for why you would)
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

USE_ROR = "--no-ror" not in sys.argv
CLASSES = ["duplicate", "hierarchy", "unrelated"]


def norm(s):
    """Lowercase, strip punctuation, collapse spaces: 'U.S. Dept.' -> 'u s dept'."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", s.lower())).strip()


# ---- ROR lookup: for each name, which ROR record it matched (if confident) and that record's relatives ----------
ror_id, ror_links = {}, {}
for line in open("data/ror_slim.jsonl", encoding="utf-8"):   # slim copy of data/ror_matches.jsonl, small enough for git
    rec = json.loads(line)
    ror_id[rec["name"]] = rec["ror_id"]
    ror_links[rec["name"]] = set(rec["links"])


# ---- the signals ("features"): every one is a number computed from the two names --------------------------------
def features(a, b, vec):
    na, nb = norm(a), norm(b)
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    f = {
        "fuzzy_sort": fuzz.token_sort_ratio(na, nb) / 100,   # word order ignored
        "fuzzy_set": fuzz.token_set_ratio(na, nb) / 100,     # 1.0 when one name's words are all inside the other's
        "fuzzy_partial": fuzz.partial_ratio(na, nb) / 100,   # best matching chunk
        "minilm_cos": float(vec[a] @ vec[b]),                # meaning similarity from the embedding model
        "contains": float(short in long_),                   # one normalized name inside the other
        "len_ratio": len(short) / max(len(long_), 1),        # 1.0 = same length, small = one name is much longer
        "extra_words": abs(len(na.split()) - len(nb.split())),
        "has_comma_diff": float(("," in a) != ("," in b)),   # "X, Library" vs "X": the sub-unit pattern
    }
    if USE_ROR:
        f["both_in_ror"] = float(a in ror_id and b in ror_id)
        f["same_ror"] = float(a in ror_id and b in ror_id and ror_id[a] == ror_id[b])
        f["ror_linked"] = float(a in ror_id and b in ror_id and (ror_id[b] in ror_links[a] or ror_id[a] in ror_links[b]))
    return f


def build(df, vec):
    return pd.DataFrame([features(a, b, vec) for a, b in zip(df.name_a, df.name_b)])


# ---- data -----------------------------------------------------------------------------------------------------------
train, test = pd.read_csv("data/pairs_train.csv"), pd.read_csv("data/pairs_test.csv")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
names = sorted(set(train.name_a) | set(train.name_b) | set(test.name_a) | set(test.name_b))
vec = dict(zip(names, embedder.encode(names, normalize_embeddings=True, batch_size=64)))

X_train, X_test = build(train, vec), build(test, vec)
y_train, y_test = train.label, test.label
print(f"{len(X_train)} train pairs, {len(X_test)} test pairs, {X_train.shape[1]} features: {list(X_train.columns)}\n")

# ---- two models: one readable, one stronger -------------------------------------------------------------------------
models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "gradient boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0),
}
for name, model in models.items():
    model.fit(X_train, y_train)                       # learning happens here: it finds how to weigh the features
    pred = model.predict(X_test)
    proba_dup = model.predict_proba(X_test)[:, list(model.classes_).index("duplicate")]
    is_dup = (y_test == "duplicate").astype(int)
    p = ((pred == "duplicate") & (is_dup == 1)).sum() / max((pred == "duplicate").sum(), 1)
    r = ((pred == "duplicate") & (is_dup == 1)).sum() / is_dup.sum()
    print(f"=== {name}   (duplicate-vs-rest: precision {p:.2f}, recall {r:.2f}, F1 {2 * p * r / max(p + r, 1e-9):.2f}, "
          f"AP {average_precision_score(is_dup, proba_dup):.2f})")
    print(classification_report(y_test, pred, labels=CLASSES, digits=2, zero_division=0))
    print("confusion matrix (rows = judge label, columns = predicted), order:", CLASSES)
    print(confusion_matrix(y_test, pred, labels=CLASSES), "\n")

lr = models["logistic regression"]
coef = pd.DataFrame(lr[-1].coef_, index=lr[-1].classes_, columns=X_train.columns).T.round(2)
print("logistic regression weights (positive = pushes toward that class; features are standardized):")
print(coef.to_string())
print("\nFor comparison on the same test set: fuzzy alone F1 0.82 / AP 0.92, off-the-shelf MiniLM alone F1 0.81 / AP 0.90.")

# ---- MISSIONS (your turn) -------------------------------------------------------------------------------------------
# 1. Add a feature. Idea: "same first word" (does the first word of both names match?). Add it to features(), rerun,
#    and check whether the numbers move. Features that do not help are as informative as ones that do.
# 2. Read the weights table. Which feature pushes hardest toward "hierarchy"? Does that match your intuition?
# 3. Run with --no-ror. How much do the ROR features contribute? Why might ROR features be risky? (Hint: how did we
#    choose which pairs to label?)
