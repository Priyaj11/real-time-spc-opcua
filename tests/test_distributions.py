"""Tests for the statistical helpers used by the simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spc_opcua.simulator.distributions import (
    exponential_approach,
    make_rng,
    normal,
    split_variance,
)


def test_same_seed_gives_the_same_numbers() -> None:
    a = make_rng(123).normal(size=50)
    b = make_rng(123).normal(size=50)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_numbers() -> None:
    a = make_rng(123).normal(size=50)
    b = make_rng(124).normal(size=50)
    assert not np.array_equal(a, b)


def test_two_generators_do_not_share_state() -> None:
    """Each generator has its own stream, so drawing from one never affects the other."""
    first = make_rng(5)
    second = make_rng(5)
    first.normal(size=100)  # consume from one only
    assert first.normal() != second.normal()


def test_normal_mean_and_spread_are_correct() -> None:
    rng = make_rng(2024)
    draws = [normal(rng, mean=10.0, std_dev=2.0) for _ in range(20_000)]
    assert float(np.mean(draws)) == pytest.approx(10.0, abs=0.05)
    assert float(np.std(draws, ddof=1)) == pytest.approx(2.0, rel=0.03)


def test_zero_std_dev_returns_the_mean_exactly() -> None:
    rng = make_rng(1)
    assert normal(rng, mean=7.5, std_dev=0.0) == 7.5


def test_zero_std_dev_consumes_no_random_numbers() -> None:
    """A constant tag must not shift the random stream for the other tags."""
    rng = make_rng(99)
    normal(rng, mean=7.5, std_dev=0.0)
    after_constant = normal(rng, mean=0.0, std_dev=1.0)

    fresh = make_rng(99)
    first_draw = normal(fresh, mean=0.0, std_dev=1.0)

    assert after_constant == first_draw


def test_warm_up_starts_at_the_starting_value() -> None:
    assert exponential_approach(0.0, start=22.0, target=42.0, tau_s=180.0) == 22.0


def test_warm_up_closes_63_percent_of_the_gap_after_one_tau() -> None:
    value = exponential_approach(180.0, start=22.0, target=42.0, tau_s=180.0)
    expected = 42.0 - 20.0 * math.exp(-1.0)  # about 34.6
    assert value == pytest.approx(expected)
    assert 0.62 < (value - 22.0) / 20.0 < 0.64


def test_warm_up_is_nearly_finished_after_five_tau() -> None:
    value = exponential_approach(900.0, start=22.0, target=42.0, tau_s=180.0)
    assert value == pytest.approx(42.0, abs=0.2)


def test_warm_up_is_monotonic_when_heating() -> None:
    previous = -math.inf
    for t in range(0, 600, 10):
        value = exponential_approach(float(t), start=22.0, target=42.0, tau_s=180.0)
        assert value > previous
        previous = value


def test_zero_tau_jumps_straight_to_target() -> None:
    assert exponential_approach(0.0, start=22.0, target=42.0, tau_s=0.0) == 42.0


def test_split_variance_removes_the_explained_part() -> None:
    # 0.5 squared is 0.25; 0.3 squared is 0.09; the remainder is 0.16, root 0.4
    assert split_variance(0.5, 0.3) == pytest.approx(0.4)


def test_split_variance_returns_zero_when_fully_explained() -> None:
    assert split_variance(0.4, 0.4) == 0.0


def test_split_variance_never_returns_negative() -> None:
    assert split_variance(0.2, 0.9) == 0.0


def test_split_variance_leaves_everything_when_nothing_explained() -> None:
    assert split_variance(0.35, 0.0) == pytest.approx(0.35)