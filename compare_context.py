"""Compare judge labels WITHOUT vs WITH ROR context on the same 600 pairs.

Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.

Usage: python compare_context.py data/judged_600.jsonl data/judged_600_ctx.jsonl
Prints label shares by pair source for both runs, how many labels changed, and the flips that changed most.
"""
import sys

import pandas as pd

plain = pd.read_json(sys.argv[1], lines=True)
ctx = pd.read_json(sys.argv[2], lines=True)
src = pd.read_csv("data/judge_sample_600.csv")[["name_a", "name_b", "source"]]
df = plain[["name_a", "name_b", "label"]].merge(ctx[["name_a", "name_b", "label"]], on=["name_a", "name_b"], suffixes=("_plain", "_ctx")).merge(src, on=["name_a", "name_b"])
print("pairs compared:", len(df))

for col in ["label_plain", "label_ctx"]:
    print(f"\n{col} (row %, by source)")
    print((pd.crosstab(df.source, df[col], normalize="index") * 100).round(0).to_string())

print(f"\nlabel changed on {(df.label_plain != df.label_ctx).sum()} of {len(df)} pairs")
print(pd.crosstab(df.label_plain, df.label_ctx, margins=True).to_string())
print("\nrandom_negative pairs that ROR context pushed away from 'unrelated' (a sign context can mislead):")
bad = df[(df.source == "random_negative") & (df.label_ctx != "unrelated") & (df.label_ctx != "disagree")]
print(bad[["name_a", "name_b", "label_plain", "label_ctx"]].to_string(index=False) if len(bad) else "  none")
