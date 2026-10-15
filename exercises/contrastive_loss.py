"""Exercise 1: write the contrastive loss in plain PyTorch.   (Your file. Only edit the function body.)

Situation: a model turned each institution name into a vector. For a batch of N name pairs you have
    emb_a   shape (N, D)   vectors for the first name of each pair
    emb_b   shape (N, D)   vectors for the second name of each pair
    labels  shape (N,)     1.0 if the pair is a duplicate, 0.0 if it is not

Compute ONE number that says how wrong the vectors are:
    distance d for a pair      = 1 - cosine_similarity(a, b)          (0 = same direction, 1 = unrelated)
    a duplicate pair (1.0)     should have a SMALL distance   -> its loss is   d squared
    a non-duplicate pair (0.0) should be at least `margin` apart -> its loss is  max(0, margin - d) squared
    the batch loss             = the MEAN of the per-pair losses

Requirements
  - Use torch operations only, so that `loss.backward()` works (no .item(), no numpy, no Python loops over the batch).
  - Return a scalar tensor (0-dimensional).
  - Do not change the function name or its arguments.

Run the checks:   python -m pytest tests/test_contrastive_loss.py -v
Goal: all green. When it is, tell Claude and we review it together (expect follow-up questions).
"""
import torch


def contrastive_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    raise NotImplementedError("write me")
