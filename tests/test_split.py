import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import split_pairs  # noqa: E402


def test_split_of_is_stable_and_valid():
    for key in ["https://ror.org/abc", "Purdue University", "Universität zu Köln"]:
        assert split_pairs.split_of(key) == split_pairs.split_of(key)
        assert split_pairs.split_of(key) in split_pairs.FRACTIONS


def test_fractions_sum_to_100():
    assert sum(split_pairs.FRACTIONS.values()) == 100


@pytest.mark.skipif(not (ROOT / "data/pairs_train.csv").exists(), reason="run split_pairs.py first")
def test_no_name_appears_in_two_splits():
    seen = {}
    for split in split_pairs.FRACTIONS:
        df = pd.read_csv(ROOT / f"data/pairs_{split}.csv")
        for name in set(df.name_a) | set(df.name_b):
            assert seen.setdefault(name, split) == split, f"{name!r} is in {seen[name]} and {split}"
