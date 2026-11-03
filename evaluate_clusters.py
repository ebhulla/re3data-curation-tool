"""Score a clustering against the judged pairs.

Usage:  python evaluate_clusters.py             (calls cluster.build_clusters on all distinct institution names)
        python evaluate_clusters.py --show 15   (also print wrong merges)

For every judged pair (train / val / test files) whose two names are both clustered, it asks "did the clustering
put them in the same cluster?" and compares with the judge's label:
  duplicate pairs   -> SHOULD be together   (share together = recall of merging)
  hierarchy pairs   -> should NOT be merged (share together = wrong merge)
  unrelated pairs   -> should NOT be merged (share together = wrong merge)
Merge precision = of all judged pairs that ended up together, the share the judge called duplicate.
Note: the decider's thresholds were tuned on train/val, so quote the TEST row as the honest number.
"""
import argparse

import pandas as pd


def to_root_map(result):
    """Accept {name: root} or {root: [names]} and return {name: root}."""
    first = next(iter(result.values()))
    if isinstance(first, (list, tuple, set)):
        return {name: root for root, members in result.items() for name in members}
    return dict(result)


def score_clusters(result, show=0):
    root_of = to_root_map(result)
    names = set(root_of)
    sizes = pd.Series(root_of).value_counts()
    print(f"names: {len(root_of)}   clusters: {sizes.size}   five largest: {sizes.head(5).tolist()}\n")

    rows, wrong = [], []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/pairs_{split}.csv")
        df = df[df.name_a.isin(names) & df.name_b.isin(names)]
        together = [root_of[a] == root_of[b] for a, b in zip(df.name_a, df.name_b)]
        df = df.assign(together=together)
        for label in ["duplicate", "hierarchy", "unrelated"]:
            part = df[df.label == label]
            rows.append({"split": split, "label": label, "pairs": len(part), "together": int(part.together.sum()),
                         "share": round(part.together.mean(), 3) if len(part) else float("nan")})
        merged = df[df.together]
        prec = (merged.label == "duplicate").mean() if len(merged) else float("nan")
        rows.append({"split": split, "label": "MERGE PRECISION", "pairs": len(merged),
                     "together": int((merged.label == "duplicate").sum()), "share": round(prec, 3)})
        wrong.append(merged[merged.label != "duplicate"].assign(split=split))

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nread: duplicate rows -> want HIGH share; hierarchy / unrelated rows -> want LOW share; merge precision -> HIGH")
    if show:
        bad = pd.concat(wrong)
        print(f"\n{len(bad)} judged non-duplicate pairs were merged; first {min(show, len(bad))}:")
        for _, r in bad.head(show).iterrows():
            print(f"  [{r.split}/{r.label}] {r.name_a[:55]} | {r.name_b[:55]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    from cluster import build_clusters

    institutions = pd.read_csv("data/institutions_clean.csv")["institution_name"].dropna().unique()
    score_clusters(build_clusters(institutions, 150), show=args.show)
