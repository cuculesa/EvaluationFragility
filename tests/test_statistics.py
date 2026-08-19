import pytest

from evalfrag.statistics import (
    benjamini_hochberg,
    exact_mcnemar,
    paired_bootstrap_delta,
    wilson_interval,
)


def test_wilson_interval_contains_observed_rate() -> None:
    lo, hi = wilson_interval(60, 100, 0.95)
    assert lo < 0.60 < hi
    assert (lo, hi) == pytest.approx((0.5020, 0.6906), abs=0.001)


def test_exact_mcnemar_uses_paired_discordances() -> None:
    only_b, only_a, p = exact_mcnemar([0, 0, 1, 1], [1, 1, 1, 0])
    assert (only_b, only_a) == (2, 1)
    assert p == 1.0


def test_paired_bootstrap_is_reproducible() -> None:
    args = dict(resamples=2000, confidence=0.95, seed=7)
    first = paired_bootstrap_delta([0, 0, 1, 1], [0, 1, 1, 1], **args)
    second = paired_bootstrap_delta([0, 0, 1, 1], [0, 1, 1, 1], **args)
    assert first == second
    assert first[0] == 0.25
    assert first[1] <= first[0] <= first[2]


def test_bh_adjustment_is_bounded_and_order_preserving() -> None:
    p = [0.001, 0.02, 0.04, 0.7]
    q = benjamini_hochberg(p)
    assert all(0 <= value <= 1 for value in q)
    assert q == pytest.approx([0.004, 0.04, 0.0533333, 0.7], rel=1e-5)
