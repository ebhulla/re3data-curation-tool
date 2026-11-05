"""Build unlabeled candidate institution-name pairs from ROR matches.

Sources: same_ror (two names -> same ROR id), ror_parent_child (ROR links the two ids),
random_negative (different ids, no ROR relationship). Labels come later from the LLM judge.
"""
import itertools
import json
import random
from collections import defaultdict

import pandas as pd

random.seed(42)
N_RANDOM = 3000

by_id = defaultdict(list)
related = set()
for line in open("data/ror_matches.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    if not rec["candidates"] or not rec["candidates"][0]["chosen"]:
        continue
    c = rec["candidates"][0]
    by_id[c["ror_id"]].append(rec["name"])
    for rel in c["relationships"]:
        if rel["type"] in ("parent", "child"):
            related.add(frozenset((c["ror_id"], rel["id"])))

rows = []
for rid, names in by_id.items():
    for a, b in itertools.combinations(sorted(names), 2):
        rows.append((a, b, "same_ror", rid, rid))
for pair in related:
    if len(pair) == 2 and all(i in by_id for i in pair):
        i, j = sorted(pair)
        for a, b in itertools.product(by_id[i], by_id[j]):
            rows.append((a, b, "ror_parent_child", i, j))

ids = list(by_id)
seen = 0
while seen < N_RANDOM:
    i, j = random.sample(ids, 2)
    if frozenset((i, j)) in related:
        continue
    rows.append((random.choice(by_id[i]), random.choice(by_id[j]), "random_negative", i, j))
    seen += 1

df = pd.DataFrame(rows, columns=["name_a", "name_b", "source", "ror_a", "ror_b"]).drop_duplicates(["name_a", "name_b"])
df.to_csv("data/candidate_pairs.csv", index=False, encoding="utf-8")
print(df.source.value_counts().to_string())
print("total", len(df), "| distinct ROR ids", len(ids))
