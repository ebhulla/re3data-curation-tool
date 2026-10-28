# Technical learning log (Ekam)

Maintained by Claude, started 2026-10-26. Levels: **Independent** = can explain and apply without help;
**Developing** = understands the explanation, needs guidance applying it; **Unfamiliar** = cannot yet explain or apply.
Levels are judged from what actually happened (evidence in brackets), and are revised as you show more.

## Concepts

| Concept | Level | Evidence / note |
|---|---|---|
| Embeddings and cosine similarity (meaning vs spelling) | Developing | Computed cosine by hand (3/3 correct); explained "meaning vs characters" but at first said fuzzy would rate "added word" pairs higher, and missed that embeddings capture topic, not part-of vs same-as |
| Precision, recall, F1, AP; reading a confusion matrix | Developing | Reads confusion matrices and chooses the next target from them; AP vs F1 (threshold-free) not yet owned |
| Train / validation / test discipline | Developing | Ran val to check generalization; accepts that test is a one-time exam; needed prompting that val becomes "used" once its mistakes drive changes |
| Data leakage (why names must not appear in two splits) | Developing | Understands the idea; the split design was mine |
| Small test sets and statistical uncertainty (F1 0.68-0.92 across splits) | Developing | Saw two early "wins" vanish across 10 splits; not yet able to size noise yourself |
| Loss functions: contrastive loss, margin, hard-pair mining | Developing | Wrote both exercises (17/17 tests); online selection took four attempts, with my pseudocode |
| Gradients, backprop, `loss.backward()`, why `.item()` cuts the graph | Unfamiliar | Seen in a toy example only; never written a training loop |
| Gradient descent, learning rate, schedules, batch size | Unfamiliar | Swept learning rate on Negishi but did not run or interpret the mechanics |
| Overfitting and regularization (train vs validation loss) | Unfamiliar | Noticed rules fit train better than test; no loss-curve work yet |
| Fine-tuning: full vs adapter (LoRA), frozen vs trainable parameters | Unfamiliar | Fine-tuning ran through a high-level Trainer; not examined |
| Tokenization and sequence length (MiniLM truncates at ~256 tokens) | Unfamiliar | Not covered |
| Label design (hierarchy lumped with unrelated at 0.0) | Developing | Found by code review; hypothesis (not proven) that it limits fine-tuning |
| Feature engineering and combining signals (3-class model) | Developing | Ran and read the weights table; did not write it |
| Rule-based normalization (replacement dictionary, stopwords) | Independent | Built both, tested with one-liners, rerun to confirm |

## Python and tooling patterns
| Pattern | Level |
|---|---|
| Boolean masks and indexing (`d[labels == 0].min()`) | Developing |
| `&` / `|` with parentheses on tensors/columns | Developing (recurring weak spot) |
| pandas `merge` with the right keys | Developing |
| Debug with focused prints / one-liners from the shell | Developing |
| Git (add by name, pull/merge, push), Slurm basics | Developing |

## Bugs and root causes (yours and mine)
- Merge on one key produced 858 rows instead of 600 -> both name columns are the key; diagnose by counting columns after the merge.
- `max`/`min` with `key=len` return the same string when lengths tie -> rule compared a name with itself (56 false duplicates).
- Dictionary replacement did nothing -> the entry was missing (twice), and the lookup was on the whole string, not per word.
- Computed `count` but never used it in a condition -> unused variables are a smell; check each computed value is used.
- Stale `import` in an open Python prompt -> test with `python -c` from the shell.
- Terminal copy dropped spaces at line breaks -> looked like a data bug; it was a paste artifact.
- Blocking cutoff compared `len(word)` (letters in the word) instead of `len(names)` (institutions containing it) -> counts jumped at cutoff 10 ("university" has 10 letters) and plateaued after 20. Found by a hand-computed upper bound (5,456 words x 10 pairs = 54,560 < 127,354 observed). Habit: when a number looks off, bound it by hand before trusting it. (2026-10-28, found by Ekam with hints)
- `finetune.py` showed a stray `import argparse1.` in the Codespace -> `git diff` before discarding anything separates a typo from real work.
- (Mine) A shell heredoc turned `\n` into a real newline and I pushed broken code; a hardcoded baseline line was stale. Verify before pushing.

## Experimental findings (with caveats)
- Substring rule fails; fuzzy and off-the-shelf MiniLM tie (~0.8 F1) with opposite errors.
- Fine-tuning MiniLM did not beat off-the-shelf in two sweeps (6 runs each); cause untested.
- Combined model did not beat fuzzy at duplicates across 10 splits; its added value is the hierarchy class.
- Own rules: train 0.84, val 0.90, test 0.80 (111 pairs; noise about +/-0.04).

## Topics to review next
1. What `loss.backward()` and `optimizer.step()` do, by writing a tiny training loop on toy data.
2. How to tell overfitting from underfitting (train vs validation curves).
3. Reading F1 vs AP; sizing uncertainty on a small test set (bootstrap).
4. Tokenization and truncation in MiniLM.
