"""Score decider.decide() against the judged pairs and show every mistake.

Usage:  python evaluate_decider.py                 (train split: use this while developing)
        python evaluate_decider.py --split test    (final check: use sparingly)
        python evaluate_decider.py --show 40       (show more mistakes)
"""
import argparse

import pandas as pd

from decider import decide

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train", choices=["train", "val", "test"])
ap.add_argument("--show", type=int, default=20, help="how many wrongly decided pairs to print")
args = ap.parse_args()

CLASSES = ["duplicate", "hierarchy", "unrelated"]
df = pd.read_csv(f"data/pairs_{args.split}.csv")
df["predicted"] = [decide(a, b) for a, b in zip(df.name_a, df.name_b)]

print(f"split: {args.split}   pairs: {len(df)}   accuracy: {(df.predicted == df.label).mean():.2f}\n")
print("per class (judge label):   precision = of the pairs you called X, how many really were X;  recall = of the real X, how many you found")
for c in CLASSES:
    called, real = (df.predicted == c), (df.label == c)
    tp = (called & real).sum()
    p, r = tp / max(called.sum(), 1), tp / max(real.sum(), 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"  {c:<10} precision {p:.2f}   recall {r:.2f}   F1 {f1:.2f}   ({real.sum()} real)")

print("\nconfusion matrix (rows = judge label, columns = your answer):")
print(pd.crosstab(df.label, df.predicted).reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_string())

bad = df[df.predicted != df.label]
print(f"\n{len(bad)} mistakes; first {min(args.show, len(bad))}:")
for _, r in bad.head(args.show).iterrows():
    print(f"  judge: {r.label:<9} you: {r.predicted:<9} | {r.name_a[:60]} | {r.name_b[:60]}")
