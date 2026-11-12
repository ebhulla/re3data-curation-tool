import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exercises.contrastive_loss import contrastive_loss  # noqa: E402


def batch(pairs, labels):
    a = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    return a, b, torch.tensor(labels, dtype=torch.float32)


def test_identical_duplicate_has_zero_loss():
    a, b, y = batch([((1.0, 0.0), (1.0, 0.0))], [1.0])
    assert contrastive_loss(a, b, y).item() == pytest.approx(0.0, abs=1e-6)


def test_orthogonal_duplicate_is_penalised_by_distance_squared():
    # cosine 0 -> distance 1 -> loss 1
    a, b, y = batch([((1.0, 0.0), (0.0, 1.0))], [1.0])
    assert contrastive_loss(a, b, y).item() == pytest.approx(1.0, abs=1e-6)


def test_far_apart_non_duplicate_has_zero_loss():
    # cosine 0 -> distance 1, already beyond margin 0.5
    a, b, y = batch([((1.0, 0.0), (0.0, 1.0))], [0.0])
    assert contrastive_loss(a, b, y).item() == pytest.approx(0.0, abs=1e-6)


def test_close_non_duplicate_is_pushed_apart():
    # cosine 0.8 -> distance 0.2 -> (0.5 - 0.2)^2 = 0.09
    a, b, y = batch([((1.0, 0.0), (0.8, 0.6))], [0.0])
    assert contrastive_loss(a, b, y).item() == pytest.approx(0.09, abs=1e-5)


def test_margin_is_respected():
    # same pair, margin 0.2 -> distance 0.2 is exactly at the margin -> loss 0
    a, b, y = batch([((1.0, 0.0), (0.8, 0.6))], [0.0])
    assert contrastive_loss(a, b, y, margin=0.2).item() == pytest.approx(0.0, abs=1e-6)


def test_batch_loss_is_the_mean_over_pairs():
    pairs = [((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (0.8, 0.6))]
    a, b, y = batch(pairs, [1.0, 0.0])          # losses 1.0 and 0.09 -> mean 0.545
    assert contrastive_loss(a, b, y).item() == pytest.approx(0.545, abs=1e-5)


def test_returns_scalar_tensor():
    a, b, y = batch([((1.0, 0.0), (0.0, 1.0))] * 3, [1.0, 0.0, 1.0])
    out = contrastive_loss(a, b, y)
    assert isinstance(out, torch.Tensor) and out.dim() == 0


def test_loss_ignores_vector_length():
    # cosine only cares about direction
    a, b, y = batch([((1.0, 0.0), (0.8, 0.6))], [0.0])
    assert contrastive_loss(a * 7, b * 0.3, y).item() == pytest.approx(0.09, abs=1e-5)


def test_gradients_flow_to_the_embeddings():
    a = torch.tensor([[1.0, 0.2], [0.3, 1.0]], requires_grad=True)
    b = torch.tensor([[0.9, 0.5], [1.0, 0.1]], requires_grad=True)
    y = torch.tensor([1.0, 0.0])
    contrastive_loss(a, b, y).backward()
    assert a.grad is not None and b.grad is not None
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_works_with_more_dimensions():
    torch.manual_seed(0)
    a, b = torch.randn(8, 384), torch.randn(8, 384)
    y = torch.tensor([1.0, 0.0] * 4)
    out = contrastive_loss(a, b, y)
    assert torch.isfinite(out)
