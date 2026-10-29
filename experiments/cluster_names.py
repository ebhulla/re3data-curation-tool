"""End-to-end demo: collapse duplicate institution names into canonical clusters and audit the result.

Steps:
  1. Score every pair of distinct names with a duplicate detector (fuzzy or MiniLM).
  2. Merge pairs scoring >= the threshold chosen on the TRAIN pairs (experiments/baselines.py) into clusters (connected components).
  3. Canonical name = the spelling used by the most repositories (ties: shorter, then alphabetical).
  4. Audit against the judge labels: wrong merges (judged hierarchy/unrelated but same cluster) and missed duplicates.

Hierarchy is NOT merged here: a sub-unit stays its own institution (parent links are future work).

Usage: python experiments/cluster_names.py fuzzy      or      python experiments/cluster_names.py minilm
Outputs: data/normalized_names_<method>.csv  (name, cluster_id, canonical, n_repos, cluster_size)
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

METHOD = sys.argv[1] if len(sys.argv) > 1 else "fuzzy"
THRESHOLD = {"fuzzy": 0.885, "minilm": 0.916}[METHOD]  # best-F1 thresholds from train (experiments/baselines.py, 2026-10-14)
KEEP = {"duplicate", "hierarchy", "unrelated"}


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", s.lower())).strip()


inst = pd.read_csv("data/institutions_clean.csv").dropna(subset=["institution_name"])
n_repos = inst.groupby("institution_name").repo_id.nunique()
names = sorted(n_repos.index)
idx = {n: i for i, n in enumerate(names)}
print(f"method={METHOD} threshold={THRESHOLD}  distinct names: {len(names)}")

# 1. pairwise scores -> edges above the threshold
if METHOD == "fuzzy":
    normed = [norm(n) for n in names]
    sim = process.cdist(normed, normed, scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1) / 100.0
else:
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(names, batch_size=64, normalize_embeddings=True)
    sim = emb @ emb.T
np.fill_diagonal(sim, 0)
ii, jj = np.where(np.triu(sim >= THRESHOLD, k=1))
print("pairs merged (edges):", len(ii))

# 2. connected components (union-find)
parent = list(range(len(names)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a, b in zip(ii, jj):
    parent[find(a)] = find(b)
roots = [find(i) for i in range(len(names))]
cluster_id = {r: k for k, r in enumerate(sorted(set(roots)))}
df = pd.DataFrame({"name": names, "cluster_id": [cluster_id[r] for r in roots], "n_repos": [int(n_repos[n]) for n in names]})
df["cluster_size"] = df.groupby("cluster_id").name.transform("size")

# 3. canonical = most-used spelling
df["_len"] = df.name.str.len()
best = df.sort_values(["cluster_id", "n_repos", "_len", "name"], ascending=[True, False, True, True]).drop_duplicates("cluster_id")
df["canonical"] = df.cluster_id.map(dict(zip(best.cluster_id, best.name)))
df = df.drop(columns="_len")
df.to_csv(f"data/normalized_names_{METHOD}.csv", index=False, encoding="utf-8")

n_clusters = df.cluster_id.nunique()
sizes = df.groupby("cluster_id").size()
print(f"\n{len(names)} distinct names -> {n_clusters} clusters ({100 * (1 - n_clusters / len(names)):.1f}% reduction)")
print(f"singletons: {(sizes == 1).sum()}   clusters with 2+ names: {(sizes > 1).sum()}   largest: {sizes.max()}   clusters of size >= 6: {(sizes >= 6).sum()}")

print("\nlargest clusters (check for chain merges):")
for cid in sizes.sort_values(ascending=False).head(5).index:
    members = df[df.cluster_id == cid].sort_values("n_repos", ascending=False)
    print(f"  [{len(members)}] canonical={members.canonical.iloc[0]!r}; e.g. {members.name.iloc[1:4].tolist()}")

print("\nrandom merges (canonical <- variant):")
multi = df[(df.cluster_size > 1) & (df.name != df.canonical)].sample(12, random_state=3)
for _, r in multi.iterrows():
    print(f"  {r.canonical!r}  <-  {r['name']!r}")

# 4. audit against judge labels
frames = [pd.read_json(p, lines=True) for p in ["data/judged_600.jsonl", "data/judged_mined.jsonl"]]
judged = pd.concat(frames, ignore_index=True)
judged = judged[judged.label.isin(KEEP)]
cid = dict(zip(df.name, df.cluster_id))
judged = judged[judged.name_a.isin(cid) & judged.name_b.isin(cid)].copy()
judged["same_cluster"] = [cid[a] == cid[b] for a, b in zip(judged.name_a, judged.name_b)]
test = pd.read_csv("data/pairs_test.csv")
test_keys = set(zip(test.name_a, test.name_b))
judged["in_test"] = [(a, b) in test_keys for a, b in zip(judged.name_a, judged.name_b)]


def audit(d, title):
    dup, hier, unrel = (d[d.label == k] for k in ("duplicate", "hierarchy", "unrelated"))
    print(f"\n{title}: {len(d)} judged pairs")
    print(f"  duplicates placed in the same cluster (recall): {dup.same_cluster.sum()}/{len(dup)} = {dup.same_cluster.mean():.0%}")
    print(f"  hierarchy pairs wrongly merged:                  {hier.same_cluster.sum()}/{len(hier)} = {hier.same_cluster.mean():.0%}")
    print(f"  unrelated pairs wrongly merged:                  {unrel.same_cluster.sum()}/{len(unrel)} = {unrel.same_cluster.mean():.0%}")


audit(judged[judged.in_test], "TEST pairs only (threshold was NOT tuned on these)")
audit(judged, "ALL judged pairs (threshold was tuned on the train part, so this is optimistic)")
