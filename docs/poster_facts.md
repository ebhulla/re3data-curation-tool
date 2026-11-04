# Poster fact sheet (draft 2026-11-04)

Every number below was produced by code in this repo. "Trust" says how far to lean on it. Quote the source if asked.

## 1. The data
| Fact | Value | Source |
|---|---|---|
| Distinct institution name strings | 6,148 (from 9,413 repository-institution links, 3,529 repositories) | `data/institutions_clean.csv` |
| Names matched confidently to ROR | 4,018 of 6,148 (65%) | `data/ror_matches.jsonl` |
| Same-ROR pairs that were true duplicates | 16% of 200 sampled (35% hierarchy, 41% unrelated) | 600-pair sample, `docs/results_2026-10-14.md` |

## 2. The answer key (AI judge)
| Fact | Value | Trust |
|---|---|---|
| Judged pairs | 3,400 (3,125 usable; 243 disagreed between name orders, about 7%) | solid |
| Agreement with 30 hand labels | 29 of 29 order-consistent pairs | weak (n=30, mostly hierarchy) |
| Random pairs called "unrelated" | 195 of 200 (98%) | decent sanity check |
| Split | train 2,176 / val 71 / **test 111** (31 duplicates, 42 hierarchy, 38 unrelated); no name in two folds | solid |

## 3. Methods compared (test set, 111 pairs, duplicate vs not)
| Method | F1 | AP |
|---|---|---|
| substring rule | 0.47 | 0.36 |
| exact (normalized) | 0.56 | 0.56 |
| fuzzy string matching | 0.82 | 0.92 |
| MiniLM embeddings, off the shelf | 0.81 | 0.90 |
| MiniLM fine-tuned (6 runs, Negishi) | 0.76-0.82 (mean 0.79) | 0.88-0.89 |
| combined 3-class model, 10 random splits | 0.78 +/- 0.06 (fuzzy alone 0.82 +/- 0.06) | 0.90 vs 0.91 |

Reading: simple string rules are as good as the AI models for duplicates; fine-tuning did not help (no evidence it does).
Trust: medium. 31 test duplicates means one pair moves recall about 3 points; the same method ranged F1 0.68-0.92 across splits.

## 4. My rule-based decider (duplicate / hierarchy / unrelated)
| Version | train acc | val acc | test acc |
|---|---|---|---|
| v1 (frozen) | 0.84 | 0.90 | **0.80** (duplicate precision 0.96 = 22/23; recall 0.71) |
| v1.1 (+ tail-aware fuzzy rule) | 0.83 | 0.92 | **not run yet** (run it once, report both versions) |

Val was partly tuned on (I read its mistakes), so test is the honest number. Rules fire in order: prefix rule, exact match, fuzzy >= 0.90 (unless the parts after the last comma differ), containment -> hierarchy, else unrelated.

## 5. Blocking: avoiding 18.9 million comparisons
| Cutoff (max names per word) | Candidate pairs | Share of 18.9M | True duplicates still compared |
|---|---|---|---|
| 50 | 85,086 | 0.45% | 88.0% |
| 100 | 208,687 | 1.1% | 97.8% |
| **150** | **269,771** | **1.4%** | **99.2% (488 of 492)** |
| 300 | 469,630 | 2.5% | 99.4% |

Trust: optimistic. Measured on pairs we labeled, which favor similar-looking names. The 4 missed pairs share no usable word (accent, acronym, translation).

## 6. Clustering result
| Version | Names -> clusters | Largest cluster |
|---|---|---|
| v1 | 6,148 -> 5,465 | 11 (four different NIH institutes merged: a flaw) |
| v1.1 (tail rule) | 6,148 -> **5,546** (9.8% fewer) | 9 (all spellings of one department: correct) |

v1.1 scored against judged pairs: train merge precision 90.1%; test merge precision 90.9% (20 of 22 merged pairs were duplicates; 1 hierarchy and 1 unrelated pair wrongly merged); true duplicates grouped: 62% train, 64.5% test.
Decider edges on the 269,771 candidates (v1.1): 776 duplicate, 3,329 hierarchy.

First full output (`data/normalized_names.csv`, 2026-11-04): 6,148 rows; 5,546 clusters; 602 names are non-canonical spellings; 1,455 names have a parent (comma-prefix rule).
**Known flaw found by a consistency check:** 4 clusters contain a name and its own parent (e.g. University of Toronto + University of Toronto Libraries), caused by the "one extra word" prefix rule plus chaining. Use as the poster's "what a check caught" example; fix = restrict the rule to legal/generic suffix words.

## Best story for the poster (one line each)
1. **Problem:** a department is not a duplicate of its university, so string-similarity merging goes wrong.
2. **Finding:** simple rules matched AI models; fine-tuning did not help; hierarchy is the hard part.
3. **Engineering:** blocking made it 70x cheaper with 99% of true pairs kept; the tail-aware rule fixed a real chain-merge flaw.
4. **Honesty:** small test sets, AI-made labels, merge precision about 90% and duplicate recall about 65%.

## Do NOT claim
- That fine-tuning "doesn't work" (only: no gain in our tests).
- That the pipeline is X% accurate overall (we measured pieces, on small sets).
- Anything about acronym/translation duplicates (under-sampled and unsolved).
- That v1.1 beats v1 on test (not measured).

## Figures worth making (in this order)
1. Pipeline diagram: names -> blocking -> decider -> clusters (+ parent links, review list).
2. Blocking curve: cutoff vs pairs and vs true duplicates kept (pick 150).
3. Method comparison bar chart with spread (section 3).
4. Before/after cluster: the NIH chain (11 names, 4 institutes) vs the fixed result.
5. Confusion matrix of the decider on test.
6. Honest-limits box.

## Still to compute (small jobs)
1. Run `python evaluate_decider.py --split test` ONCE for v1.1 and report both versions.
2. Bootstrap confidence interval for the test accuracy (error bar for the poster).
3. Cluster-level audit: sample 30 merged pairs from the final clusters and check them by eye.
