"""Build the labeled pair dataset and split it into train / val / test without leakage.

Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.

Leakage guard: each NAME is assigned to a split by a stable hash of its group, where the group is its
ROR id (names matched to the same organization stay together) or the name itself if unmatched.
A pair is kept only if both names land in the same split, so no name and no ROR cluster ever appears
in two splits. Pairs that straddle splits are dropped (reported below).

Labels come from the judge. Kept: duplicate, hierarchy, unrelated, only when both swap orders agreed.
Dropped: disagree, uncertain, error.

Outputs: data/pairs_{train,val,test}.csv  with columns name_a, name_b, label, is_duplicate, source_file
"""
import hashlib
import json
import os

import pandas as pd

JUDGED_FILES = ["data/judged_600.jsonl", "data/judged_mined.jsonl"]  # judged_mined may not exist yet
KEEP = {"duplicate", "hierarchy", "unrelated"}
FRACTIONS = {"train": 80, "val": 10, "test": 10}  # percent, must sum to 100


def load_ror_groups():
    group = {}
    for line in open("data/ror_matches.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        cands = rec["candidates"]
        if cands and cands[0]["chosen"]:
            group[rec["name"]] = cands[0]["ror_id"]
    return group


def split_of(group_key):
    bucket = int(hashlib.md5(group_key.encode("utf-8")).hexdigest(), 16) % 100
    cutoff = 0
    for name, pct in FRACTIONS.items():
        cutoff += pct
        if bucket < cutoff:
            return name


def main():
    groups = load_ror_groups()
    frames = []
    for path in JUDGED_FILES:
        if os.path.exists(path):
            df = pd.read_json(path, lines=True)
            df["source_file"] = os.path.basename(path)
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print("judged pairs:", len(data))
    print(data.label.value_counts().to_string())

    data = data[data.label.isin(KEEP)].copy()
    data["split_a"] = [split_of(groups.get(n, n)) for n in data.name_a]
    data["split_b"] = [split_of(groups.get(n, n)) for n in data.name_b]
    same = data.split_a == data.split_b
    print(f"\nkept after label filter: {len(data)}; dropped for straddling splits: {(~same).sum()}")
    data = data[same].copy()
    data["split"] = data.split_a
    data["is_duplicate"] = (data.label == "duplicate").astype(int)

    cols = ["name_a", "name_b", "label", "is_duplicate", "source_file"]
    for name in FRACTIONS:
        part = data[data.split == name]
        part[cols].to_csv(f"data/pairs_{name}.csv", index=False, encoding="utf-8")
    print("\n" + pd.crosstab(data.split, data.label, margins=True).to_string())


if __name__ == "__main__":
    main()
