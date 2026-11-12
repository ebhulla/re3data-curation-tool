"""Draw the human-audit samples for the poster. Run from the repo root:  python experiments/audit_sample.py

Writes three sheets to audit/ with a blank `human_label` column for you to fill in by hand:
  A_merges.csv      random pairs the tool put in the same cluster      -> merge precision
  B_candidates.csv  for random names, every lookalike the tool merged or any simple method would merge -> recall + baselines
  C_parents.csv     random parent links the tool produced             -> parent accuracy

Label vocabulary (write exactly one of these words):
  A and B:  same       = the two names are the same institution (a spelling variant, abbreviation or translation)
            part       = one is a unit inside the other (department, library, program)
            different  = two different institutions
  C:        correct / wrong  (is the parent really the institution above this unit?)

The seed is fixed, so everyone gets the same sample. Do not look at the tool's answer column before labeling if you can
avoid it (the sheets are shuffled so the answer is hard to infer). Developed with AI assistance (Claude); see docs/AUTHORSHIP.md.
"""
import os
import random
import sys
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.getcwd())
from cluster import candidate_pairs, cluster  # noqa: E402
from decider import norm  # noqa: E402

SEED = 7
N_A, N_B, N_C = (int(x) for x in (sys.argv[1:4] if len(sys.argv) >= 4 else (100, 30, 50)))
FUZZY = 0.885   # the fuzzy baseline's threshold chosen on train (experiments/baselines.py)
OUT = "audit"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(SEED)

table = pd.read_csv("data/normalized_names.csv")
cid = dict(zip(table.name, table.cluster_id))
members = table.groupby("cluster_id").name.apply(list).to_dict()

# ---- A: pairs the tool merged
merged_pairs = [p for g in members.values() if len(g) > 1 for p in combinations(sorted(g), 2)]
sample_a = rng.sample(merged_pairs, min(N_A, len(merged_pairs)))
a = pd.DataFrame([{"name_a": x, "name_b": y, "human_label": "", "notes": ""} for x, y in sample_a]).sample(frac=1, random_state=SEED)
a.insert(0, "pair_id", [f"A{i + 1:03d}" for i in range(len(a))])
a.to_csv(f"{OUT}/A_merges.csv", index=False, encoding="utf-8")

# ---- B: per-name candidate lists; every method is scored on the same candidates
names = table.name.tolist()
index = cluster(names)
blocked = candidate_pairs(index, 150)
neighbors = {}
for x, y in blocked:
    neighbors.setdefault(x, set()).add(y)
    neighbors.setdefault(y, set()).add(x)


def lookalikes(name):
    near = sorted(neighbors.get(name, ()), key=lambda o: -fuzz.token_sort_ratio(norm(name), norm(o)))[:5]
    mates = [m for m in members[cid[name]] if m != name]
    return list(dict.fromkeys(mates + near))


eligible = [n for n in names if lookalikes(n)]
rows = []
for k, name in enumerate(rng.sample(eligible, min(N_B, len(eligible)))):
    for other in lookalikes(name):
        rows.append({"name_id": f"B{k + 1:02d}", "name": name, "candidate": other, "human_label": "", "notes": "",
                     "tool_same_cluster": cid[name] == cid[other],
                     "exact_after_cleaning": norm(name) == norm(other),
                     "fuzzy_ge_threshold": fuzz.token_sort_ratio(norm(name), norm(other)) / 100 >= FUZZY})
b = pd.DataFrame(rows)
b.to_csv(f"{OUT}/B_candidates.csv", index=False, encoding="utf-8")

# ---- C: parent links
links = table.dropna(subset=["parent_name"])
sample_c = links.sample(n=min(N_C, len(links)), random_state=SEED)
c = pd.DataFrame({"link_id": [f"C{i + 1:03d}" for i in range(len(sample_c))], "unit": sample_c.name.values,
                  "parent": sample_c.parent_name.values, "human_label": "", "notes": ""})
c.to_csv(f"{OUT}/C_parents.csv", index=False, encoding="utf-8")

print(f"A: {len(a)} merged pairs (of {len(merged_pairs)})   B: {b.name_id.nunique()} names, {len(b)} candidate rows")
print(f"C: {len(c)} links (of {len(links)})")
print("labeling effort (A and C ~20s per row, B ~8s per row):", round(((len(a) + len(c)) * 20 + len(b) * 8) / 60), "minutes")
