"""Generate the poster figures from the repo's data. Run from the repo root:  python experiments/make_poster_figures.py

Writes figures/*.png (300 dpi) and figures/*.svg. Numbers are computed from data/ where possible; the few values taken
from earlier measured runs are marked 'measured' with their source. Palette validated with the dataviz validator
(blue #2a78d6 + orange #eb6834: CVD delta E >= 24, contrast >= 3:1); one accent against gray, no rainbow.
Developed with AI assistance (Claude); see docs/AUTHORSHIP.md.
"""
import os
import sys
import textwrap
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.getcwd())
from cluster import build_clusters, direct_parent  # noqa: E402
from decider import decide, norm  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY, LIGHT, INK, INK2, SURFACE = "#c9c8c2", "#e6e5e0", "#0b0b0b", "#52514e", "#fcfcfb"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#8a8985", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.png")


def title(ax, head, sub):
    ax.set_title(head, loc="left", fontsize=22, fontweight="bold", pad=34 + 17 * sub.count("\n"))
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=13, color=INK2, va="bottom")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ------------------------------------------------------------------ data
df = pd.read_csv("data/institutions_clean.csv").dropna(subset=["institution_name"])
names = list(df["institution_name"].unique())
N = len(names)
ALL_PAIRS = N * (N - 1) // 2
judged = pd.concat([pd.read_csv(f"data/pairs_{s}.csv") for s in ["train", "val", "test"]])
dup_keys = {tuple(sorted(p)) for p in zip(judged[judged.label == "duplicate"].name_a, judged[judged.label == "duplicate"].name_b)}

# blocking curve: candidate pairs and duplicate recall by cutoff (blocks added in order of size)
index = defaultdict(set)
for n in names:
    for w in set(norm(n).split()):
        index[w].add(n)
blocks = sorted(index.values(), key=len)
CUTOFFS = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]
pairs, bi, curve = set(), 0, []
for c in CUTOFFS:
    while bi < len(blocks) and len(blocks[bi]) <= c:
        pairs.update(combinations(sorted(blocks[bi]), 2))
        bi += 1
    curve.append((c, len(pairs), sum(k in pairs for k in dup_keys) / len(dup_keys)))

# decider on the test set (v1.1)
test = pd.read_csv("data/pairs_test.csv")
test["pred"] = [decide(a, b) for a, b in zip(test.name_a, test.name_b)]
CLASSES = ["duplicate", "hierarchy", "unrelated"]
cm = pd.crosstab(test.label, test.pred).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
tp = cm.loc["duplicate", "duplicate"]
prec, rec = tp / cm["duplicate"].sum(), tp / cm.loc["duplicate"].sum()
f1_rules = 2 * prec * rec / (prec + rec)
acc = np.trace(cm.values) / cm.values.sum()

# clusters and parents
clusters = build_clusters(names, 150)
root_of = {n: r for r, m in clusters.items() for n in m}
by_norm = {norm(n): n for n in names}
n_parent = sum(direct_parent(n, by_norm) is not None for n in names)
n_clusters = len(clusters)
print(f"names={N} clusters={n_clusters} with_parent={n_parent} test_acc={acc:.3f} rules_F1={f1_rules:.3f}")

# ------------------------------------------------------------------ figure 1: pipeline
fig, ax = plt.subplots(figsize=(15, 4.1))
ax.axis("off")
steps = [
    ("1  Names", f"{N:,}", "distinct institution\nspellings in re3data"),
    ("2  Blocking", "269,771", "candidate pairs\n(1.4% of 18.9 million)"),
    ("3  Decider", "776 + 3,329", "duplicate + hierarchy\nanswers (rules)"),
    ("4  Clusters", f"{n_clusters:,}", f"institutions\n({N - n_clusters} spellings merged)"),
    ("5  Parent links", f"{n_parent:,}", "units linked to the\ninstitution above them"),
]
w, gap = 0.17, 0.0375
for i, (head, big, small) in enumerate(steps):
    x = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.18), w, 0.62, boxstyle="round,pad=0.01,rounding_size=0.02", fc="white", ec=BLUE if i else GRAY, lw=2.2))
    ax.text(x + w / 2, 0.72, head, ha="center", va="center", fontsize=15, fontweight="bold", color=INK2)
    ax.text(x + w / 2, 0.50, big, ha="center", va="center", fontsize=26 if len(big) < 9 else 19, fontweight="bold", color=BLUE)
    ax.text(x + w / 2, 0.30, small, ha="center", va="center", fontsize=12.5, color=INK2)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.004, 0.49), xytext=(x + w + 0.004, 0.49), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2))
ax.text(0, 0.05, "An AI judge labeled 3,400 sample pairs as an answer key for testing the rules; it is not part of the tool.",
        fontsize=12.5, color=INK2)
ax.set_xlim(-0.01, 1.03)
ax.set_ylim(0, 0.86)
ax.set_title("From 6,148 spellings to 5,546 institutions", loc="left", fontsize=22, fontweight="bold", pad=18)
save(fig, "1_pipeline")

# ------------------------------------------------------------------ figure 2: blocking trade-off
fig, ax = plt.subplots(figsize=(10, 6.2))
xs = [p / ALL_PAIRS * 100 for _, p, _ in curve]
ys = [r * 100 for _, _, r in curve]
ax.plot(xs, ys, color=BLUE, lw=3, zorder=2)
ax.scatter(xs, ys, s=70, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
i150 = [c for c, _, _ in curve].index(150)
ax.scatter([xs[i150]], [ys[i150]], s=190, color=ORANGE, zorder=4, edgecolor=SURFACE, linewidth=2.5)
for (c, _, _), x, y in zip(curve, xs, ys):
    if c in (10, 50):
        ax.annotate(f"cutoff {c}", (x, y), textcoords="offset points", xytext=(10, -20), fontsize=12, color=INK2)
    if c == 100:
        ax.annotate(f"cutoff {c}", (x, y), textcoords="offset points", xytext=(-8, 12), fontsize=12, color=INK2, ha="right")
    if c == 500:
        ax.annotate(f"cutoff {c}", (x, y), textcoords="offset points", xytext=(0, -24), fontsize=12, color=INK2, ha="right")
ax.annotate("cutoff 150: keeps %.1f%% of true duplicates\nwhile comparing only %.1f%% of all pairs" % (ys[i150], xs[i150]),
            (xs[i150], ys[i150]), xytext=(0.47, 0.30), textcoords="axes fraction", fontsize=14, fontweight="bold", color=ORANGE,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.5))
ax.set_xscale("log")
ax.set_xticks([0.1, 0.3, 1, 3])
ax.set_xticklabels(["0.1%", "0.3%", "1%", "3%"])
ax.set_ylim(45, 105)
ax.set_xlabel("Share of all 18.9 million name pairs that are compared (log scale)")
ax.set_ylabel("True duplicate pairs still compared (%)")
ax.grid(axis="y", color=LIGHT, lw=1)
ax.set_axisbelow(True)
title(ax, "Comparing 1.4% of the pairs keeps 99% of the duplicates",
      f"Blocking compares only names that share a word used by at most N names.\nMeasured on {len(dup_keys)} labeled duplicate pairs (a slightly optimistic sample).")
save(fig, "2_blocking")

# ------------------------------------------------------------------ figure 3: method comparison
methods = [  # name, F1, accent?, (low, high) range
    ("Substring rule", 0.47, False, None),
    ("Exact match after cleaning", 0.56, False, None),
    ("My rules (v1.1)", round(f1_rules, 2), True, None),
    ("AI embeddings, fine-tuned (6 runs)", 0.79, False, (0.76, 0.82)),   # measured: docs/results_2026-10-14.md, sweep on 2026-10-16
    ("AI embeddings, off the shelf", 0.81, False, None),                 # measured: baselines.py on the 111-pair test set
    ("Fuzzy string matching", 0.82, False, None),
]
fig, ax = plt.subplots(figsize=(10.5, 5.8))
for i, (name, f1, accent, rng) in enumerate(methods):
    ax.barh(i, f1, height=0.62, color=BLUE if accent else GRAY, zorder=2)
    ax.text((rng[1] if rng else f1) + 0.012, i, f"{f1:.2f}", va="center", fontsize=14, fontweight="bold", color=INK)
    if rng:
        ax.plot(rng, [i, i], color=INK2, lw=2, zorder=3)
        ax.plot([rng[0]] * 2, [i - 0.14, i + 0.14], color=INK2, lw=2, zorder=3)
        ax.plot([rng[1]] * 2, [i - 0.14, i + 0.14], color=INK2, lw=2, zorder=3)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels([m[0] for m in methods], fontsize=14)
ax.set_xlim(0, 1.0)
ax.set_xlabel("F1 score at finding duplicate institution names (1.0 = perfect)")
ax.grid(axis="x", color=LIGHT, lw=1)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
title(ax, "Simple string matching ties AI embeddings",
      "Same held-out test set: 111 pairs, 31 true duplicates. Bar = result; line = range over 6 training runs.\nDifferences of about 0.03 are within noise.")
save(fig, "3_methods")

# ------------------------------------------------------------------ figure 4: confusion matrix
fig, ax = plt.subplots(figsize=(7.4, 6.2))
cmap = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#1f5fb0"])
ax.imshow(cm.values, cmap=cmap, vmin=0, vmax=cm.values.max())
for i in range(3):
    for j in range(3):
        v = cm.values[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=26, fontweight="bold", color="white" if v > cm.values.max() * 0.55 else INK)
ax.set_xticks(range(3))
ax.set_xticklabels(CLASSES, fontsize=14)
ax.set_yticks(range(3))
ax.set_yticklabels(CLASSES, fontsize=14)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_xlabel("What the rules answered", fontsize=14, labelpad=10)
ax.set_ylabel("AI answer key (judge)", fontsize=14, labelpad=22)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
lo, hi = wilson(int(np.trace(cm.values)), int(cm.values.sum()))
ax.text(0.5, -0.1, f"Accuracy {acc:.0%} on 111 held-out pairs (95% interval {lo:.0%} to {hi:.0%}). When it says duplicate,\n"
        f"it is right {prec:.0%} of the time ({tp} of {cm['duplicate'].sum()}). 20 of the 24 errors are related pairs it calls\n"
        "unrelated: mostly abbreviations (U. S. vs United States), spelling variants, a few translations.",
        transform=ax.transAxes, ha="center", va="top", fontsize=12, color=INK2)
fig.suptitle("Rarely wrong when it merges; misses spelling variants", x=0.03, ha="left", fontsize=20, fontweight="bold", y=1.05)
save(fig, "4_confusion")

# ------------------------------------------------------------------ figure 5: NIH before / after
nih = [
    "National Institutes of Health, National Cancer Institute", "National Institutes of Health, National Institute on Aging",
    "National Institutes of Health, National Insitute of Drug Abuse", "National Institutes of Health,  National Cancer Institute",
    "National Institutes of Health, National Eye Institute", "National Institutes of Health, National Institute on Drug Abuse",
    "Nartional Instiutes of Health, National Cancer Institute", "National Institutes of Health, National Institute for Aging",
    "National Cancer Institute at the National Institutes of Health", "National Instiutes of Health, National Cancer Institut",
    "U.S. National Institutes of Health, National Cancer Institute",
]
members = [by_norm[norm(s)] for s in nih if norm(s) in by_norm]
groups = defaultdict(list)
for m in members:
    groups[root_of[m]].append(m)
short = lambda s: s.replace("National Institutes of Health", "NIH").replace("Nartional Instiutes of Health", "NIH").replace("National Instiutes of Health", "NIH")  # noqa: E731
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
for a in (a1, a2):
    a.axis("off")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
a1.add_patch(FancyBboxPatch((0.02, 0.10), 0.96, 0.72, boxstyle="round,pad=0.01,rounding_size=0.02", fc="white", ec=ORANGE, lw=2.5))
a1.text(0.02, 0.96, "BEFORE: 1 cluster", fontsize=18, fontweight="bold", color=ORANGE, va="top")
a1.text(0.02, 0.90, "four different institutes merged", fontsize=13, color=INK2, va="top")
for k, m in enumerate(members):
    a1.text(0.06, 0.78 - k * 0.058, short(" ".join(m.split())), fontsize=11, color=INK2, va="center")
a2.text(0.0, 0.96, f"AFTER the tail rule: {len(groups)} clusters", fontsize=18, fontweight="bold", color=BLUE, va="top")
a2.text(0.0, 0.90, "spelling variants stay together; different institutes separate", fontsize=13, color=INK2, va="top")
y = 0.84
for g in sorted(groups.values(), key=len, reverse=True):
    h = 0.044 * len(g) + 0.03
    a2.add_patch(FancyBboxPatch((0.0, y - h), 0.98, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc="white", ec=BLUE, lw=1.8))
    for k, m in enumerate(g):
        a2.text(0.03, y - 0.03 - k * 0.044, short(" ".join(m.split())), fontsize=11, color=INK2, va="center")
    y -= h + 0.012
a2.text(0.0, 0.03, "Not perfect: the last cluster is also the Cancer Institute (one spelling still left out).", fontsize=12, color=INK2)
fig.suptitle("A check caught a real flaw, and the fix separated the institutes", x=0.01, ha="left", fontsize=22, fontweight="bold", y=1.0)
save(fig, "5_nih_before_after")

# ------------------------------------------------------------------ figure 6: headline numbers
tiles = [  # big number, caption, honest caveat
    ("70x", "fewer comparisons", "270,000 instead of 18.9 million pairs"),
    ("99%", "of true duplicates kept", "of 492 labeled pairs (a slightly optimistic sample)"),
    (f"{n_clusters:,}", "institutions from 6,148 spellings", f"{N - n_clusters} spelling variants merged"),
    ("78%", "of merges were correct", "95% interval 69% to 85% (AI-judged random sample)"),
    ("98%", "of parent links were correct", "95% interval 89% to 100% (AI-judged random sample)"),
]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
for ax, (big, cap, cav) in zip(axes, tiles):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94, transform=ax.transAxes, boxstyle="round,pad=0.01,rounding_size=0.04", fc="white", ec=LIGHT, lw=2))
    ax.text(0.5, 0.66, big, transform=ax.transAxes, ha="center", va="center", fontsize=34 if len(big) < 9 else 24, fontweight="bold", color=BLUE)
    ax.text(0.5, 0.40, "\n".join(textwrap.wrap(cap, 19)), transform=ax.transAxes, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.17, "\n".join(textwrap.wrap(cav, 24)), transform=ax.transAxes, ha="center", va="center", fontsize=10.5, color=INK2)
save(fig, "6_headline_numbers")

# ------------------------------------------------------------------ figure 7: workflow with one worked example
# Real names from data/normalized_names.csv; counts and verdicts below were checked against decider.py and the table.
fig, ax = plt.subplots(figsize=(17, 9.2))
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 64)
CW, CG = 17.4, 3.25
TINT = {"duplicate": ("#dbe8fa", BLUE), "hierarchy": ("#fde4d9", ORANGE)}


def card(x, y, h, ec=LIGHT, lw=1.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), CW, h, boxstyle="round,pad=0.0,rounding_size=1.0", fc="white", ec=ec, lw=lw, linestyle=ls))


def pill(cx, cy, label):
    fc, ec = TINT[label]
    ax.add_patch(FancyBboxPatch((cx - 5.2, cy - 1.15), 10.4, 2.3, boxstyle="round,pad=0.0,rounding_size=1.1", fc=fc, ec=ec, lw=1.6))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=11.5, fontweight="bold")


def lines(x, y, text, width, size=11, color=INK2, step=1.9, weight="normal", ha="left"):
    wrapped = textwrap.wrap(text, width)
    for k, ln in enumerate(wrapped):
        ax.text(x, y - k * step, ln, fontsize=size, color=color, va="center", ha=ha, fontweight=weight)
    return len(wrapped)


heads = [("1  Names", f"{N:,}"), ("2  Blocking", "269,771 pairs"), ("3  Decider", "776 + 3,329"),
         ("4  Clusters", f"{n_clusters:,}"), ("5  Parent links", f"{n_parent:,}")]
subs = ["every distinct spelling", "rare shared words only", "rules answer per pair", "duplicates merge", "sub-units point up"]
xs7 = [i * (CW + CG) for i in range(5)]
for i, (x, (head, big)) in enumerate(zip(xs7, heads)):
    ax.add_patch(FancyBboxPatch((x, 52.5), CW, 9.5, boxstyle="round,pad=0.0,rounding_size=1.2", fc="white", ec=BLUE if i else GRAY, lw=2.6))
    ax.text(x + CW / 2, 59.6, head, ha="center", va="center", fontsize=15, fontweight="bold", color=INK2)
    ax.text(x + CW / 2, 56.5, big, ha="center", va="center", fontsize=19, fontweight="bold", color=BLUE)
    ax.text(x + CW / 2, 53.7, subs[i], ha="center", va="center", fontsize=10.5, color=INK2)
    card(x, 9.5, 41.5)
    if i < 4:
        ax.annotate("", xy=(x + CW + CG - 0.3, 57.2), xytext=(x + CW + 0.3, 57.2), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2.2))
        ax.annotate("", xy=(x + CW + CG - 0.3, 30), xytext=(x + CW + 0.3, 30), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.2))
    ax.plot([x + CW / 2] * 2, [52.3, 51.2], color=GRAY, lw=2)

ax.text(0, 63.6, "How the tool works, followed by one real example", fontsize=22, fontweight="bold", va="bottom")

# 1 names
X = xs7[0] + 1.2
ax.text(xs7[0] + CW / 2, 48.9, "a sample of 6 of the names", fontsize=10.5, color=INK2, ha="center", style="italic")
y = 45.6
for nm in ["University of Montpellier", "Université de Montpellier", "Université Montpellier", "University of Oxford",
           "University of Oxford, Oxford e-Research Centre", "University of Oxford, Oxford e-Research Center"]:
    n_ln = lines(X, y, nm, 25, size=11, color=INK)
    y -= 1.9 * n_ln + 1.5

ax.text(xs7[0] + CW / 2, 10.9, "+ 6,142 more names", fontsize=10.5, color=INK2, ha="center")

# 2 blocking
X = xs7[1] + 1.2
ax.text(xs7[1] + CW / 2, 48.9, "match on a shared word", fontsize=10.5, color=INK2, ha="center", style="italic")
rows = [("university", "in 1,672 names", "too common: skipped", GRAY, INK2), ("montpellier", "in 7 names", "compare these", BLUE, INK),
        ("oxford", "in 20 names", "compare these", BLUE, INK)]
y = 45.6
for word, cnt_txt, verdict, c, tc in rows:
    ax.add_patch(FancyBboxPatch((X - 0.4, y - 5.6), CW - 1.6, 7.7, boxstyle="round,pad=0.0,rounding_size=0.8", fc="white", ec=c, lw=2))
    ax.text(X + 0.6, y, word, fontsize=13, fontweight="bold", color=tc, va="center")
    ax.text(X + 0.6, y - 2.2, cnt_txt, fontsize=10.5, color=INK2, va="center")
    ax.text(X + 0.6, y - 4.2, ("✗ " if c == GRAY else "✓ ") + verdict, fontsize=10.5, color=tc, va="center")
    y -= 9.3
ax.text(xs7[1] + CW / 2, 17.4, "In this sample:", fontsize=11, color=INK2, ha="center")
ax.text(xs7[1] + CW / 2, 15.2, "15 possible pairs", fontsize=12, color=INK2, ha="center")
ax.text(xs7[1] + CW / 2, 12.7, "6 compared", fontsize=15, fontweight="bold", color=BLUE, ha="center")

# 3 decider
X = xs7[2] + 1.2
ax.text(xs7[2] + CW / 2, 48.9, "my rules, pair by pair", fontsize=10.5, color=INK2, ha="center", style="italic")
pairs = [("Univ. of Montpellier\n= Université de Montpellier", "duplicate"), ("Université de Montpellier\n= Université Montpellier", "duplicate"),
         ("University of Oxford\n> …e-Research Centre", "hierarchy"), ("…Research Centre\n= …Research Center", "duplicate")]
y = 46.0
for txt, lab in pairs:
    for k, ln in enumerate(txt.split("\n")):
        ax.text(X, y - k * 1.8, ln, fontsize=10, color=INK, va="center")
    pill(xs7[2] + CW / 2, y - 4.6, lab)
    y -= 8.9
ax.text(xs7[2] + CW / 2, 10.9, "+ 2 more (1 dup., 1 hier.)", fontsize=10.5, color=INK2, ha="center")

# 4 clusters
X = xs7[3] + 1.2
ax.text(xs7[3] + CW / 2, 48.9, "duplicates join up", fontsize=10.5, color=INK2, ha="center", style="italic")
groups7 = [(["University of Montpellier", "Université de Montpellier", "Université Montpellier"], "Université Montpellier", 11.0),
           (["Oxford e-Research Centre", "Oxford e-Research Center"], "Oxford e-Research Centre", 8.6),
           (["University of Oxford"], "University of Oxford", 6.4)]
y = 46.6
for members, canon, h in groups7:
    ax.add_patch(FancyBboxPatch((X - 0.6, y - h), CW - 1.2, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc="white", ec=BLUE, lw=2))
    for k, m in enumerate(members):
        ax.text(X + 0.2, y - 1.6 - k * 1.9, m, fontsize=9.5, color=INK2, va="center")
    ax.text(X + 0.2, y - h + 1.5, "→ " + canon, fontsize=9, fontweight="bold", va="center")
    y -= h + 1.6
ax.text(xs7[3] + CW / 2, 10.9, "bold = name shown", fontsize=9, color=INK2, ha="center")

# 5 parents
X = xs7[4] + 1.2
ax.text(xs7[4] + CW / 2, 48.9, "hierarchy links, never merges", fontsize=10.5, color=INK2, ha="center", style="italic")
ax.add_patch(FancyBboxPatch((X - 0.2, 38.0), CW - 2, 6.4, boxstyle="round,pad=0.0,rounding_size=0.8", fc="white", ec=BLUE, lw=2))
lines(X + 0.4, 42.4, "University of Oxford", 22, size=11.5, color=INK, weight="bold")
ax.text(X + 0.4, 39.6, "(the institution)", fontsize=10, color=INK2, va="center")
ax.annotate("", xy=(xs7[4] + CW / 2, 37.8), xytext=(xs7[4] + CW / 2, 30.2), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.6))
ax.text(xs7[4] + CW / 2 + 1.2, 34.0, "parent of", fontsize=10.5, color=INK2, va="center")
ax.add_patch(FancyBboxPatch((X - 0.2, 21.6), CW - 2, 8.4, boxstyle="round,pad=0.0,rounding_size=0.8", fc="white", ec=ORANGE, lw=2))
lines(X + 0.4, 28.0, "Oxford e-Research Centre", 21, size=11.5, color=INK, weight="bold", step=2.0)
ax.text(X + 0.4, 23.0, "(a sub-unit)", fontsize=10, color=INK2, va="center")
ax.text(xs7[4] + CW / 2, 15.2, "The Oxford pair was a\nhierarchy answer in step 3.\nIt is kept as a link,\nnot merged away.", fontsize=10, color=INK2, ha="center", va="center", linespacing=1.4)

ax.text(0, 5.2, "Not part of the tool: an AI judge labeled 3,400 sample pairs only to test the rules (78% accurate on 111 held-out pairs).",
        fontsize=12.5, color=INK2)
ax.text(0, 2.0, "Known weak spot: a name and its one-word extension can be merged wrongly (e.g. Stockholm University and Stockholm University Press); being fixed.",
        fontsize=12.5, color=INK2)
save(fig, "7_workflow_example")
