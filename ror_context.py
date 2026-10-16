"""Add ROR registry facts to each pair, so the judge can be run with and without them.

Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.

Inspired by Sundaram et al. 2024 (PMID 40417549): an LLM grounded in a structured knowledge base
does better than an LLM alone. Here the knowledge base is ROR.

Usage: python ror_context.py data/judge_sample_600.csv data/judge_sample_600_ctx.csv
Adds two columns, ctx_ab and ctx_ba: the same facts phrased for A/B order and for the swapped order.
"""
import json
import sys

import pandas as pd


def load_ror():
    match = {}
    for line in open("data/ror_matches.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        c = rec["candidates"][0] if rec["candidates"] else None
        if c and c["chosen"]:
            match[rec["name"]] = c
    return match


def relation(x, y):
    """Describe how y's ROR record relates to x's, or '' if unrelated."""
    if x["ror_id"] == y["ror_id"]:
        return "both names matched the SAME ROR record"
    for rel in x["relationships"]:
        if rel["id"] == y["ror_id"]:
            return f"ROR lists the second record as a {rel['type']} of the first"
    return ""


def describe(first, second, match):
    cx, cy = match.get(first), match.get(second)
    parts = [
        f"first name -> ROR: {cx['ror_name']!r}" if cx else "first name -> no reliable ROR match",
        f"second name -> ROR: {cy['ror_name']!r}" if cy else "second name -> no reliable ROR match",
    ]
    if cx and cy:
        parts.append(relation(cx, cy) or "ROR records are different and not directly linked")
    return "ROR facts: " + "; ".join(parts)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    match = load_ror()
    df = pd.read_csv(src)
    df["ctx_ab"] = [describe(a, b, match) for a, b in zip(df.name_a, df.name_b)]
    df["ctx_ba"] = [describe(b, a, match) for a, b in zip(df.name_a, df.name_b)]
    df.to_csv(dst, index=False, encoding="utf-8")
    print(df.ctx_ab.str.contains("SAME").sum(), "same-record,", df.ctx_ab.str.contains("lists the second").sum(), "linked,",
          df.ctx_ab.str.contains("no reliable").sum(), "with an unmatched name, of", len(df))
