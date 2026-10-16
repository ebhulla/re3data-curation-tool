import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exercises.online_contrastive_loss import online_contrastive_loss  # noqa: E402


def pair_with_cosine(c):
    """Two unit vectors in 2-D whose cosine similarity is c."""
    return (1.0, 0.0), (c, math.sqrt(1 - c * c))


def make(cosines, labels):
    pairs = [pair_with_cosine(c) for c in cosines]
    a = torch.tensor([p[0] for p in pairs], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.float32, requires_grad=True)
    return a, b, torch.tensor(labels, dtype=torch.float32)


# Worked example from the docstring: distances A 0.05, B 0.40, C 0.90, D 0.30  ->  cosines 0.95, 0.60, 0.10, 0.70
EXAMPLE = ([0.95, 0.60, 0.10, 0.70], [1.0, 1.0, 0.0, 0.0])


def test_worked_example_keeps_only_b_and_d():
    a, b, y = make(*EXAMPLE)
    assert online_contrastive_loss(a, b, y).item() == pytest.approx(0.10, abs=1e-5)


def test_easy_pairs_receive_no_gradient():
    a, b, y = make(*EXAMPLE)
    online_contrastive_loss(a, b, y).backward()
    assert a.grad[0].abs().sum() == 0 and a.grad[2].abs().sum() == 0      # pairs A and C were dropped
    assert a.grad[1].abs().sum() > 0 and a.grad[3].abs().sum() > 0        # pairs B and D were kept


def test_nothing_kept_returns_zero_and_backward_works():
    a, b, y = make([1.0, 0.0], [1.0, 0.0])      # duplicate at d=0, non-duplicate at d=1: nothing to learn
    out = online_contrastive_loss(a, b, y)
    assert out.item() == pytest.approx(0.0, abs=1e-6)
    out.backward()
    assert torch.isfinite(a.grad).all()


def test_no_negatives_falls_back_to_all_pairs():
    a, b, y = make([0.95, 0.60], [1.0, 1.0])    # d = 0.05 and 0.40 -> (0.0025 + 0.16) / 2
    assert online_contrastive_loss(a, b, y).item() == pytest.approx(0.08125, abs=1e-5)


def test_no_positives_falls_back_to_all_pairs():
    a, b, y = make([0.10, 0.70], [0.0, 0.0])    # d = 0.90 and 0.30 -> (0 + 0.04) / 2
    assert online_contrastive_loss(a, b, y).item() == pytest.approx(0.02, abs=1e-5)


def test_margin_argument_is_used():
    a, b, y = make(*EXAMPLE)
    # margin 0.2: kept B (0.16) and D (max(0, 0.2 - 0.3)^2 = 0) -> 0.08
    assert online_contrastive_loss(a, b, y, margin=0.2).item() == pytest.approx(0.08, abs=1e-5)


def test_returns_finite_scalar_on_random_batch():
    torch.manual_seed(1)
    a, b = torch.randn(16, 384), torch.randn(16, 384)
    y = (torch.arange(16) % 2).float()
    out = online_contrastive_loss(a, b, y)
    assert out.dim() == 0 and torch.isfinite(out)
