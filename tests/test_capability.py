"""Tests for process capability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spc_opcua.config import TagSpec, load_config
from spc_opcua.simulator.faults import (
    FaultSchedule,
    MeanShift,
    ToolWear,
    VarianceInflation,
)
from spc_opcua.simulator.offline import bore_subgroups
from spc_opcua.spc.capability import (
    MINIMUM_ACCEPTABLE_CPK,
    Capability,
    capability_from_subgroups,
    normal_tail,
    rolling_capability,
)
from spc_opcua.spc.constants import constants_for
from spc_opcua.spc.subgroups import Subgroup

D2_FIVE = constants_for(5).d2

BORE = TagSpec(
    name="BoreDiameter",
    units="mm",
    nominal=20.0,
    std_dev=0.012,
    lsl=19.95,
    usl=20.05,
)


def constant_range_subgroups(
    mean: float, sigma: float, count: int = 30
) -> list[Subgroup]:
    """Subgroups with an exactly known mean and an exactly known R-bar.

    Every subgroup is (m-2d, m-d, m, m+d, m+2d), so its mean is exactly m and
    its range is exactly 4d. Choosing d = sigma * d2 / 4 makes R-bar / d2 come
    out at exactly sigma, which lets the capability arithmetic be checked by
    hand rather than approximately.
    """
    d = sigma * D2_FIVE / 4.0
    values = (mean - 2 * d, mean - d, mean, mean + d, mean + 2 * d)
    return [Subgroup(i, "BoreDiameter", values) for i in range(count)]


# --------------------------------------------------------------------------
# The arithmetic, checked by hand
# --------------------------------------------------------------------------


def test_cp_is_the_tolerance_over_six_sigma() -> None:
    """Tolerance 0.100, sigma 0.010, so 0.100 / 0.060 is 1.667."""
    cap = capability_from_subgroups(
        constant_range_subgroups(mean=20.0, sigma=0.010), BORE
    )
    assert cap.sigma_within == pytest.approx(0.010)
    assert cap.cp == pytest.approx(0.100 / (6 * 0.010))


def test_cpu_and_cpl_measure_the_room_to_each_limit() -> None:
    cap = capability_from_subgroups(
        constant_range_subgroups(mean=20.01, sigma=0.010), BORE
    )
    assert cap.cpu == pytest.approx((20.05 - 20.01) / (3 * 0.010))
    assert cap.cpl == pytest.approx((20.01 - 19.95) / (3 * 0.010))


def test_cpk_is_the_worse_of_the_two_sides() -> None:
    cap = capability_from_subgroups(
        constant_range_subgroups(mean=20.01, sigma=0.010), BORE
    )
    assert cap.cpk == pytest.approx(min(cap.cpu, cap.cpl))
    assert cap.cpk == pytest.approx(cap.cpu)  # closer to the upper limit


def test_a_perfectly_centred_process_has_cp_equal_to_cpk() -> None:
    cap = capability_from_subgroups(
        constant_range_subgroups(mean=20.0, sigma=0.010), BORE
    )
    assert cap.cp == pytest.approx(cap.cpk)
    assert cap.centring_loss == pytest.approx(0.0, abs=1e-9)


def test_cpk_is_never_greater_than_cp() -> None:
    """Being off centre can only cost you capability, never gain it."""
    for offset in (-0.02, -0.01, 0.0, 0.005, 0.02):
        cap = capability_from_subgroups(
            constant_range_subgroups(mean=20.0 + offset, sigma=0.010), BORE
        )
        assert cap.cpk <= cap.cp + 1e-12


def test_a_process_sitting_on_a_specification_limit_has_zero_cpk() -> None:
    cap = capability_from_subgroups(
        constant_range_subgroups(mean=19.95, sigma=0.010), BORE
    )
    assert cap.cpk == pytest.approx(0.0, abs=1e-9)
    assert cap.verdict == "NOT CAPABLE"


def test_a_process_outside_the_specification_has_negative_cpk() -> None:
    """Cpk below zero is meaningful: the average part is already scrap."""
    cap = capability_from_subgroups(
        constant_range_subgroups(mean=19.94, sigma=0.010), BORE
    )
    assert cap.cpk < 0.0


# --------------------------------------------------------------------------
# Cp versus Cpk: the distinction that matters
# --------------------------------------------------------------------------


def test_moving_the_mean_leaves_cp_alone_and_drops_cpk() -> None:
    """Cp is about the spread only. Cpk is about the spread and the position."""
    centred = capability_from_subgroups(
        constant_range_subgroups(mean=20.0, sigma=0.010), BORE
    )
    offset = capability_from_subgroups(
        constant_range_subgroups(mean=20.02, sigma=0.010), BORE
    )
    assert offset.cp == pytest.approx(centred.cp)
    assert offset.cpk < centred.cpk


def test_widening_the_spread_drops_both() -> None:
    tight = capability_from_subgroups(
        constant_range_subgroups(mean=20.0, sigma=0.010), BORE
    )
    wide = capability_from_subgroups(
        constant_range_subgroups(mean=20.0, sigma=0.020), BORE
    )
    assert wide.cp == pytest.approx(tight.cp / 2.0)
    assert wide.cpk == pytest.approx(tight.cpk / 2.0)


def test_centring_is_zero_when_centred_and_one_at_a_limit() -> None:
    centred = capability_from_subgroups(
        constant_range_subgroups(mean=20.0, sigma=0.010), BORE
    )
    at_limit = capability_from_subgroups(
        constant_range_subgroups(mean=20.05, sigma=0.010), BORE
    )
    assert centred.centring == pytest.approx(0.0)
    assert at_limit.centring == pytest.approx(1.0)


def test_the_centring_loss_is_cp_minus_cpk() -> None:
    cap = capability_from_subgroups(
        constant_range_subgroups(mean=20.015, sigma=0.010), BORE
    )
    assert cap.centring_loss == pytest.approx(cap.cp - cap.cpk)


# --------------------------------------------------------------------------
# One-sided tolerances
# --------------------------------------------------------------------------


def test_a_one_sided_tolerance_has_no_cp() -> None:
    """Cp needs two limits to have a tolerance width to divide by."""
    cycle_time = TagSpec(
        name="CycleTime", units="s", nominal=12.0, std_dev=0.25, lsl=None, usl=14.0
    )
    groups = [
        Subgroup(i, "CycleTime", (11.5, 11.75, 12.0, 12.25, 12.5)) for i in range(10)
    ]
    cap = capability_from_subgroups(groups, cycle_time)
    assert cap.cp is None
    assert cap.pp is None
    assert cap.cpl is None
    assert cap.cpu is not None
    assert cap.cpk == pytest.approx(cap.cpu)


def test_a_one_sided_tolerance_has_no_centring_measure() -> None:
    temperature = TagSpec(
        name="Temperature", units="degC", nominal=42.0, std_dev=1.2, usl=60.0
    )
    groups = [
        Subgroup(i, "Temperature", (40.0, 41.0, 42.0, 43.0, 44.0)) for i in range(10)
    ]
    cap = capability_from_subgroups(groups, temperature)
    assert cap.centring is None
    assert cap.centring_loss is None
    assert not cap.is_two_sided


def test_a_tag_with_no_limits_has_no_capability() -> None:
    """ScrapCount is counted, not measured. It belongs on a p-chart."""
    scrap = load_config().tag("ScrapCount")
    groups = [Subgroup(0, "ScrapCount", (0.0, 0.0, 1.0, 1.0, 2.0))]
    with pytest.raises(ValueError, match="no specification limits"):
        capability_from_subgroups(groups, scrap)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_capability_needs_at_least_one_subgroup() -> None:
    with pytest.raises(ValueError, match="at least one subgroup"):
        capability_from_subgroups([], BORE)


def test_mixed_subgroup_sizes_are_rejected() -> None:
    mixed = [
        Subgroup(0, "BoreDiameter", (20.0, 20.0, 20.0)),
        Subgroup(1, "BoreDiameter", (20.0, 20.0)),
    ]
    with pytest.raises(ValueError, match="same size"):
        capability_from_subgroups(mixed, BORE)


def test_a_perfectly_repeatable_process_reports_infinite_capability() -> None:
    """Zero spread fits any tolerance. Infinity beats dividing by zero."""
    groups = [Subgroup(i, "BoreDiameter", (20.0,) * 5) for i in range(5)]
    cap = capability_from_subgroups(groups, BORE)
    assert cap.sigma_within == 0.0
    assert math.isinf(cap.cpk)
    assert cap.expected_ppm_defective == 0.0


# --------------------------------------------------------------------------
# Predicted defect rate
# --------------------------------------------------------------------------


def test_the_normal_tail_matches_known_values() -> None:
    assert normal_tail(0.0) == pytest.approx(0.5)
    assert normal_tail(1.0) == pytest.approx(0.15866, abs=1e-5)
    assert normal_tail(3.0) == pytest.approx(0.0013499, abs=1e-7)
    assert normal_tail(4.0) == pytest.approx(3.167e-5, rel=1e-3)


def make_capability(cpk: float) -> Capability:
    """A centred two-sided capability with a chosen Cpk, for tail arithmetic."""
    return Capability(
        tag="X",
        sample_size=100,
        subgroup_count=20,
        mean=0.0,
        sigma_within=1.0,
        sigma_overall=1.0,
        lsl=-3.0 * cpk,
        usl=3.0 * cpk,
        cp=cpk,
        cpu=cpk,
        cpl=cpk,
        cpk=cpk,
        pp=cpk,
        ppk=cpk,
    )


def test_a_cpk_of_one_predicts_about_2700_defects_per_million() -> None:
    """Three sigma each side of a centred normal process."""
    assert make_capability(1.0).expected_ppm_defective == pytest.approx(2700, rel=0.01)


def test_the_industry_floor_predicts_about_63_defects_per_million() -> None:
    assert make_capability(1.33).expected_ppm_defective == pytest.approx(63, rel=0.05)


def test_a_cpk_of_two_predicts_almost_nothing() -> None:
    assert make_capability(2.0).expected_ppm_defective < 0.01


def test_the_predicted_rate_falls_as_capability_rises() -> None:
    rates = [make_capability(c).expected_ppm_defective for c in (0.8, 1.0, 1.33, 1.67)]
    assert all(later < earlier for earlier, later in zip(rates, rates[1:]))


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cpk,expected",
    [
        (0.5, "NOT CAPABLE"),
        (0.99, "NOT CAPABLE"),
        (1.0, "MARGINAL"),
        (1.32, "MARGINAL"),
        (1.33, "CAPABLE"),
        (1.66, "CAPABLE"),
        (1.67, "GOOD"),
        (1.99, "GOOD"),
        (2.0, "EXCELLENT"),
    ],
)
def test_the_verdict_follows_the_usual_bands(cpk: float, expected: str) -> None:
    assert make_capability(cpk).verdict == expected


def test_acceptability_is_the_1_33_floor() -> None:
    assert not make_capability(1.32).is_acceptable
    assert make_capability(MINIMUM_ACCEPTABLE_CPK).is_acceptable


# --------------------------------------------------------------------------
# Short-term versus long-term: Cpk against Ppk
# --------------------------------------------------------------------------


def test_a_stable_process_has_almost_no_gap_between_cpk_and_ppk() -> None:
    cap = capability_from_subgroups(bore_subgroups(FaultSchedule(), 1, 40), BORE)
    assert abs(cap.stability_gap) < 0.15


def test_tool_wear_opens_a_gap_between_cpk_and_ppk() -> None:
    """Short-term spread is fine. The machine is wandering between subgroups."""
    wear = FaultSchedule([ToolWear(tag="BoreDiameter", rate_per_hour=-0.05)], seed=1)
    cap = capability_from_subgroups(bore_subgroups(wear, 1, 40), BORE)
    assert cap.sigma_overall > 1.2 * cap.sigma_within
    assert cap.stability_gap > 0.15
    assert cap.ppk < cap.cpk


def test_variance_inflation_widens_both_sigmas_together() -> None:
    """This is a spread problem, not a drift problem, so the gap stays small."""
    noisy = FaultSchedule([VarianceInflation(tag="BoreDiameter", factor=2.0)], seed=1)
    cap = capability_from_subgroups(bore_subgroups(noisy, 1, 40), BORE)
    assert cap.sigma_overall == pytest.approx(cap.sigma_within, rel=0.2)
    assert abs(cap.stability_gap) < 0.15


# --------------------------------------------------------------------------
# What each fault does, on real simulated data
# --------------------------------------------------------------------------


def test_the_healthy_process_is_capable() -> None:
    cap = capability_from_subgroups(bore_subgroups(FaultSchedule(), 1, 40), BORE)
    assert cap.is_acceptable


def test_a_mean_shift_ruins_cpk_while_cp_holds() -> None:
    healthy = capability_from_subgroups(bore_subgroups(FaultSchedule(), 1, 40), BORE)
    shifted = capability_from_subgroups(
        bore_subgroups(
            FaultSchedule([MeanShift(tag="BoreDiameter", shift_sigma=2.0)], seed=1),
            1,
            40,
        ),
        BORE,
    )
    assert shifted.cp == pytest.approx(healthy.cp, rel=0.1)
    assert shifted.cpk < 0.6 * healthy.cpk
    assert not shifted.is_acceptable


def test_variance_inflation_ruins_both() -> None:
    healthy = capability_from_subgroups(bore_subgroups(FaultSchedule(), 1, 40), BORE)
    noisy = capability_from_subgroups(
        bore_subgroups(
            FaultSchedule([VarianceInflation(tag="BoreDiameter", factor=2.0)], seed=1),
            1,
            40,
        ),
        BORE,
    )
    assert noisy.cp < 0.6 * healthy.cp
    assert noisy.cpk < 0.6 * healthy.cpk


# --------------------------------------------------------------------------
# How much you can trust a capability number
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_estimator_is_unbiased_across_many_samples() -> None:
    """Averaged over forty independent 200-part samples, Cp lands on the truth."""
    theoretical_cp = 0.100 / (6 * 0.012)  # 1.389 from the configured sigma
    estimates = [
        capability_from_subgroups(bore_subgroups(FaultSchedule(), seed, 40), BORE).cp
        for seed in range(1, 41)
    ]
    assert float(np.mean(estimates)) == pytest.approx(theoretical_cp, rel=0.05)

@pytest.mark.slow 
def test_a_single_capability_number_carries_real_uncertainty() -> None:
    """Two hundred parts gives a Cp with a standard deviation near 0.1.

    This is why quoting a Cpk without a sample size means very little, and why
    a process measured at 1.35 has not really been shown to clear a 1.33 floor.
    """
    estimates = [
        capability_from_subgroups(bore_subgroups(FaultSchedule(), seed, 40), BORE).cp
        for seed in range(1, 41)
    ]
    spread = float(np.std(estimates, ddof=1))
    assert 0.05 < spread < 0.2


# --------------------------------------------------------------------------
# Rolling capability, for the dashboard trend
# --------------------------------------------------------------------------


def test_a_rolling_window_gives_one_point_per_position() -> None:
    groups = bore_subgroups(FaultSchedule(), 1, 30)
    trend = rolling_capability(groups, BORE, window=20)
    assert len(trend) == 30 - 20 + 1


def test_too_few_subgroups_gives_an_empty_trend() -> None:
    groups = bore_subgroups(FaultSchedule(), 1, 10)
    assert rolling_capability(groups, BORE, window=20) == []


def test_a_window_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 subgroups"):
        rolling_capability(bore_subgroups(FaultSchedule(), 1, 30), BORE, window=1)


def test_rolling_cpk_falls_as_a_tool_wears() -> None:
    """This is the trend line the operator watches."""
    wear = FaultSchedule([ToolWear(tag="BoreDiameter", rate_per_hour=-0.05)], seed=1)
    trend = rolling_capability(bore_subgroups(wear, 1, 60), BORE, window=20)
    assert len(trend) > 10
    assert trend[-1].cpk < trend[0].cpk - 0.5
    assert trend[0].is_acceptable is False or trend[-1].is_acceptable is False


def test_rolling_cpk_stays_flat_on_a_healthy_process() -> None:
    trend = rolling_capability(bore_subgroups(FaultSchedule(), 1, 60), BORE, window=20)
    values = [c.cpk for c in trend]
    assert max(values) - min(values) < 0.5


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def test_the_row_carries_every_index() -> None:
    row = capability_from_subgroups(
        constant_range_subgroups(mean=20.0, sigma=0.010), BORE
    ).as_row()
    for key in ("cp", "cpk", "pp", "ppk", "ppm", "verdict"):
        assert key in row


def test_the_description_names_the_two_sigmas() -> None:
    text = capability_from_subgroups(
        constant_range_subgroups(mean=20.0, sigma=0.010), BORE
    ).describe()
    assert "Sigma within" in text
    assert "Sigma overall" in text
    assert "Cpk" in text


def test_the_description_handles_a_one_sided_tolerance() -> None:
    temperature = TagSpec(
        name="Temperature", units="degC", nominal=42.0, std_dev=1.2, usl=60.0
    )
    groups = [
        Subgroup(i, "Temperature", (40.0, 41.0, 42.0, 43.0, 44.0)) for i in range(10)
    ]
    text = capability_from_subgroups(groups, temperature).describe()
    assert "n/a" in text  # Cp has no meaning here