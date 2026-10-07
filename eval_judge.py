"""Compare judge labels with the 30 hand labels: agreement, confusion matrix, disagreements."""
import json
import sys

import pandas as pd

gold = pd.read_csv("data/gold_pairs_30.csv").rename(columns={"Name A": "name_a", "Name B": "name_b", "Label": "gold"})
gold["gold"] = gold.gold.str.lower()
judged = pd.DataFrame(map(json.loads, open(sys.argv[1], encoding="utf-8")))
df = gold.merge(judged, on=["name_a", "name_b"])
print(f"{len(df)} of {len(gold)} gold pairs judged")
print("swap-consistent (both orders agree):", (df.label != "disagree").sum())
ok = df[df.label != "disagree"]
print(f"agreement with gold on consistent pairs: {(ok.label == ok.gold).mean():.0%} ({(ok.label == ok.gold).sum()}/{len(ok)})")
print(pd.crosstab(df.gold, df.label))
for _, r in df[df.label != df.gold].iterrows():
    print(f"\n[{r.ID}] gold={r.gold} judge={r.label} ({r.label_ab}/{r.label_ba})\n  A: {r.name_a}\n  B: {r.name_b}\n  why: {r.reason_ab}")
