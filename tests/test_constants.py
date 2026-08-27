"""Tests for the SPC chart constants.

The point of these tests is that the constants are DERIVED from d2 and d3 in
the code, and checked here against the values printed in the standard tables.
If a digit of d2 or d3 were mistyped, the derived A2, D3 or D4 would stop
matching the published table and these tests would fail.
"""

from __future__ import annotations

import math

import pytest

from spc_opcua.spc.constants import (
    MAX_SUBGROUP_SIZE,
    MIN_SUBGROUP_SIZE,
    constants_for,
    sigma_from_mean_range,
)

# The published table, straight out of any SPC reference. n: (A2, D3, D4).
PUBLISHED = {
    2: (1.880, 0.000, 3.267),
    3: (1.023, 0.000, 2.574),
    4: (0.729, 0.000, 2.282),
    5: (0.577, 0.000, 2.114),
    6: (0.483, 0.000, 2.004),
    7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864),
    9: (0.337, 0.184, 1.816),
    10: (0.308, 0.223, 1.777),
    12: (0.266, 0.283, 1.717),
    15: (0.223, 0.347, 1.653),
    20: (0.180, 0.415, 1.585),
    25: (0.153, 0.459, 1.541),
}


# --------------------------------------------------------------------------
# The derived constants must match the printed table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", sorted(PUBLISHED))
def test_derived_a2_matches_the_published_table(n: int) -> None:
    expected = PUBLISHED[n][0]
    assert constants_for(n).a2 == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("n", sorted(PUBLISHED))
def test_derived_d3_matches_the_published_table(n: int) -> None:
    expected = PUBLISHED[n][1]
    assert constants_for(n).d3_lower == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("n", sorted(PUBLISHED))
def test_derived_d4_matches_the_published_table(n: int) -> None:
    expected = PUBLISHED[n][2]
    assert constants_for(n).d4_upper == pytest.approx(expected, abs=0.001)


# --------------------------------------------------------------------------
# Where the constants come from
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(MIN_SUBGROUP_SIZE, MAX_SUBGROUP_SIZE + 1))
def test_a2_is_three_sigma_of_a_subgroup_mean(n: int) -> None:
    """A2 = 3 / (d2 * sqrt(n)), because a mean of n values is sqrt(n) steadier."""
    c = constants_for(n)
    assert c.a2 == pytest.approx(3.0 / (c.d2 * math.sqrt(n)))


@pytest.mark.parametrize("n", range(MIN_SUBGROUP_SIZE, MAX_SUBGROUP_SIZE + 1))
def test_the_range_limits_are_three_sigma_of_the_range(n: int) -> None:
    """D4 = 1 + 3 d3 / d2, and D3 is its mirror image, floored at zero."""
    c = constants_for(n)
    three_sigma = 3.0 * c.d3 / c.d2
    assert c.d4_upper == pytest.approx(1.0 + three_sigma)
    assert c.d3_lower == pytest.approx(max(0.0, 1.0 - three_sigma))


@pytest.mark.parametrize("n", range(2, 7))
def test_small_subgroups_have_no_lower_range_limit(n: int) -> None:
    """For n of 6 or less, three sigma of the range exceeds R-bar itself."""
    assert constants_for(n).d3_lower == 0.0


@pytest.mark.parametrize("n", range(7, 26))
def test_larger_subgroups_do_have_a_lower_range_limit(n: int) -> None:
    assert constants_for(n).d3_lower > 0.0


def test_a2_shrinks_as_subgroups_grow() -> None:
    """Bigger subgroups give steadier means, so the X-bar limits tighten."""
    values = [constants_for(n).a2 for n in range(2, 26)]
    assert all(later < earlier for earlier, later in zip(values, values[1:]))


def test_d2_grows_as_subgroups_grow() -> None:
    """More values means a wider expected range."""
    values = [constants_for(n).d2 for n in range(2, 26)]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))


# --------------------------------------------------------------------------
# Lookup behaviour
# --------------------------------------------------------------------------


def test_the_constants_report_their_own_subgroup_size() -> None:
    assert constants_for(5).n == 5


def test_a_subgroup_size_below_two_has_no_constants() -> None:
    with pytest.raises(ValueError, match="No standard SPC constants"):
        constants_for(1)


def test_a_subgroup_size_above_twentyfive_has_no_constants() -> None:
    with pytest.raises(ValueError, match="No standard SPC constants"):
        constants_for(26)


def test_the_error_message_says_which_sizes_work() -> None:
    with pytest.raises(ValueError, match="2 to 25"):
        constants_for(100)


# --------------------------------------------------------------------------
# Estimating sigma from the average range
# --------------------------------------------------------------------------


def test_sigma_is_the_mean_range_divided_by_d2() -> None:
    assert sigma_from_mean_range(2.326, n=5) == pytest.approx(1.0)


def test_a_wider_average_range_means_a_wider_process() -> None:
    assert sigma_from_mean_range(0.05, 5) > sigma_from_mean_range(0.02, 5)


def test_a_zero_range_means_a_perfectly_repeatable_process() -> None:
    assert sigma_from_mean_range(0.0, 5) == 0.0


def test_a_negative_mean_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        sigma_from_mean_range(-0.01, 5)


def test_the_same_range_implies_a_smaller_sigma_for_a_bigger_subgroup() -> None:
    """A range of 1 across 10 values means a tighter process than across 2."""
    assert sigma_from_mean_range(1.0, 10) < sigma_from_mean_range(1.0, 2)