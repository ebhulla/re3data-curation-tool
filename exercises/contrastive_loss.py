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
Goal: all green.
"""
import torch

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return (a_norm * b_norm).sum(dim=1)

#What we Did in this function:
# 1. Compute the distance `d` for each pair using cosine similarity.
# 2. Compute the loss for positive pairs (duplicates) as `d^2`.
# 3. Compute the loss for negative pairs (non-duplicates) as `max(0, margin - d)^2`.
# 4. Combine the losses using the labels.
# 5. Return the mean loss over the batch.

def contrastive_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    d = 1 - cosine_similarity(emb_a, emb_b)
    loss_pos = d ** 2
    loss_neg = torch.clamp(margin - d, min=0) ** 2
    loss = labels * loss_pos + (1 - labels) * loss_neg
    return loss.mean()
