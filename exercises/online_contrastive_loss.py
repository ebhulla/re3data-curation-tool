"""Exercise 2: online hard-pair selection on top of your contrastive loss.   (Edit only the function body.)

Same inputs as Exercise 1: emb_a (N, D), emb_b (N, D), labels (N,) with 1.0 = duplicate, 0.0 = not a duplicate.

Per-pair loss, distance d = 1 - cosine:      duplicate:  d squared        non-duplicate:  max(0, margin - d) squared

New: only the pairs the model is currently CONFUSED about contribute.
    closest_negative  = smallest d among the label-0 pairs
    farthest_positive = largest  d among the label-1 pairs
    keep a duplicate      only if its d is GREATER than closest_negative
    keep a non-duplicate  only if its d is SMALLER than farthest_positive
    return the MEAN of the per-pair loss over the KEPT pairs

Worked example (distances): A dup 0.05, B dup 0.40, C non-dup 0.90, D non-dup 0.30
    closest_negative = 0.30, farthest_positive = 0.40  ->  kept: B and D   ->  (0.16 + 0.04) / 2 = 0.10

Edge cases (the tests check each):
  - If the batch has no duplicates, or no non-duplicates, skip the selection and use ALL pairs.
  - If nothing is kept, return 0.0, and `loss.backward()` must still work (no crash, no NaN).
  - Return a scalar tensor. No .item(), no numpy, no Python loops over the batch.

Run:  python -m pytest tests/test_online_contrastive_loss.py -v      Goal: all green.
"""
from itertools import count

import torch

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return (a_norm * b_norm).sum(dim=1)

def online_contrastive_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    d = 1 - cosine_similarity(emb_a, emb_b)
    loss_pos = d ** 2
    loss_neg = torch.clamp(margin - d, min=0) ** 2

    closest_negative = d[labels == 0].min() if (labels == 0).any() else torch.tensor(float('inf'))
    farthest_positive = d[labels == 1].max() if (labels == 1).any() else torch.tensor(float('-inf'))

    per_pair_loss = labels * loss_pos + (1 - labels) * loss_neg

    if (closest_negative == float('inf')) or (farthest_positive == float('-inf')):
        keep = torch.ones_like(d, dtype=torch.bool)
    else:
        keep = (labels == 1) & (d > closest_negative) | (labels == 0) & (d < farthest_positive)

    total =  (per_pair_loss * keep).sum()
    count = keep.float().sum().clamp(min=1)
    #loss = (labels * loss_pos + (1 - labels) * loss_neg)[keep]
    return total / count
