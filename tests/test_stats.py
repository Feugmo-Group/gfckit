"""Tests for the two small statistics the JES manuscript quotes.

These exist because the published Wilson intervals and McNemar p-value were
originally computed in a throwaway command, which meant the paper carried
numbers no script could regenerate.
"""
import numpy as np

from gfckit.mechanism_id import mcnemar_exact, wilson_interval


def test_wilson_matches_published_intervals():
    """The two intervals quoted in the manuscript, to the printed precision."""
    lo, hi = wilson_interval(9, 24)             # capacity only, 37.5%
    assert round(lo * 100, 1) == 21.2
    assert round(hi * 100, 1) == 57.3
    lo, hi = wilson_interval(5, 24)             # capacity + dV/dQ, 20.8%
    assert round(lo * 100, 1) == 9.2
    assert round(hi * 100, 1) == 40.5


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    """The reason for preferring Wilson to Wald: 0/n and n/n must not escape
    [0,1], which the normal approximation does."""
    for k, n in [(0, 24), (24, 24), (1, 3), (3, 3)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_brackets_the_point_estimate():
    for k in range(0, 25):
        lo, hi = wilson_interval(k, 24)
        assert lo <= k / 24 <= hi


def test_mcnemar_reproduces_the_published_p_value():
    """4 cells only capacity classified correctly, 0 the other way: the exact
    two-sided binomial p is 2 * (1/2)^4 = 0.125."""
    a = np.array([1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)   # capacity correct
    b = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)   # capacity+dV/dQ
    nb, nc, p = mcnemar_exact(a, b)
    assert (nb, nc) == (4, 0)
    assert abs(p - 0.125) < 1e-12


def test_mcnemar_is_symmetric_in_its_arguments():
    a = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
    b = np.array([0, 1, 1, 1, 0, 1], dtype=bool)
    nb, nc, p1 = mcnemar_exact(a, b)
    nc2, nb2, p2 = mcnemar_exact(b, a)
    assert (nb, nc) == (nb2, nc2)
    assert abs(p1 - p2) < 1e-12


def test_mcnemar_no_discordant_pairs_is_p_one():
    a = np.array([1, 0, 1, 0], dtype=bool)
    nb, nc, p = mcnemar_exact(a, a)
    assert (nb, nc) == (0, 0)
    assert p == 1.0


def test_mcnemar_balanced_discordance_is_not_significant():
    """Equal numbers each way is the null: p must be 1."""
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([0, 0, 1, 1], dtype=bool)
    nb, nc, p = mcnemar_exact(a, b)
    assert (nb, nc) == (2, 2)
    assert abs(p - 1.0) < 1e-12
