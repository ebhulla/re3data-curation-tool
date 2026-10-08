"""Embed every distinct institution name and mine candidate pairs from nearest neighbors.

Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.

Why: random pairs are ~99% unrelated, so labeling them wastes judge budget. Nearest neighbors in
embedding space are exactly the pairs that look alike, which is where duplicates, hierarchies and
confusable (hard negative) pairs live.

Outputs:
  data/name_embeddings.npy   one 384-d vector per name (row order = data/names.txt)
  data/names.txt             the distinct names
  data/mined_pairs.csv       unlabeled pairs, with cosine similarity and a similarity bin
"""
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8          # neighbors per name
MIN_SIM = 0.60     # ignore weak neighbors
PER_BIN = {"0.9+": 700, "0.8-0.9": 900, "0.7-0.8": 700, "0.6-0.7": 500}  # 2,800 total, ~2 days of free Groq
SEED = 7

names = sorted(pd.read_csv("data/institutions_clean.csv").institution_name.dropna().unique())
with open("data/names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(names))

model = SentenceTransformer(MODEL)
emb = model.encode(names, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
np.save("data/name_embeddings.npy", emb)
print("embeddings:", emb.shape)

# Vectors are unit length, so a dot product IS the cosine similarity.
sim = emb @ emb.T
np.fill_diagonal(sim, -1)  # a name is not its own neighbor

pairs = {}
for i in range(len(names)):
    for j in np.argsort(-sim[i])[:TOP_K]:
        if sim[i, j] >= MIN_SIM:
            a, b = sorted((names[i], names[j]))
            pairs[(a, b)] = float(sim[i, j])

df = pd.DataFrame([(a, b, s) for (a, b), s in pairs.items()], columns=["name_a", "name_b", "cos_sim"])
print("neighbor pairs >=", MIN_SIM, ":", len(df))

# Drop pairs that are already judged (either order).
seen = set()
for path in ["data/judged_600.jsonl", "data/judged_gold.jsonl"]:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        seen.add(tuple(sorted((r["name_a"], r["name_b"]))))
df = df[[tuple(sorted(p)) not in seen for p in zip(df.name_a, df.name_b)]]
print("after removing already-judged:", len(df))

df["sim_bin"] = pd.cut(df.cos_sim, [0.6, 0.7, 0.8, 0.9, 1.01], right=False, labels=["0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9+"])
print(df.sim_bin.value_counts().sort_index().to_string())
df.to_csv("data/mined_pairs_all.csv", index=False, encoding="utf-8")

# Stratified pick, weighted toward high similarity where duplicates concentrate.
picked = pd.concat([g.sample(min(len(g), PER_BIN[str(b)]), random_state=SEED) for b, g in df.groupby("sim_bin", observed=True)])
picked = picked.sample(frac=1, random_state=SEED)
picked.to_csv("data/mined_pairs.csv", index=False, encoding="utf-8")
print("picked:", len(picked))
