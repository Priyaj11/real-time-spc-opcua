"""Tests for the X-bar and R control charts."""

from __future__ import annotations

import pytest

from spc_opcua.config import load_config
from spc_opcua.simulator.faults import (
    FaultSchedule,
    MeanShift,
    ToolWear,
    VarianceInflation,
)
from spc_opcua.simulator.machine import MachineSimulator
from spc_opcua.spc.constants import constants_for
from spc_opcua.spc.control_charts import (
    ChartPoint,
    ControlLimits,
    XbarRChart,
    compute_limits,
)
from spc_opcua.spc.subgroups import Subgroup, subgroups_from_values

# Four subgroups of five, chosen so the arithmetic can be done by hand:
# every range is exactly 8, and the means are 6, 5, 7, 6, averaging 6.
HAND_WORKED = [
    Subgroup(0, "X", (2.0, 4.0, 6.0, 8.0, 10.0)),
    Subgroup(1, "X", (1.0, 3.0, 5.0, 7.0, 9.0)),
    Subgroup(2, "X", (3.0, 5.0, 7.0, 9.0, 11.0)),
    Subgroup(3, "X", (2.0, 4.0, 6.0, 8.0, 10.0)),
]


def bore_subgroups(faults: FaultSchedule, seed: int, count: int) -> list[Subgroup]:
    """Run the simulator offline and split its bore measurements into subgroups."""
    config = load_config()
    n = config.subgroup_size
    simulator = MachineSimulator(config, seed=seed, faults=faults)
    values: list[float] = []
    while len(values) < count * n:
        sample = simulator.step()
        if sample.part_completed:
            values.append(sample.values["BoreDiameter"])
    return subgroups_from_values(values, n, tag="BoreDiameter")


# --------------------------------------------------------------------------
# The arithmetic, done by hand
# --------------------------------------------------------------------------


def test_the_centre_lines_are_the_grand_mean_and_the_mean_range() -> None:
    limits = compute_limits(HAND_WORKED)
    assert limits.grand_mean == pytest.approx(6.0)
    assert limits.mean_range == pytest.approx(8.0)
    assert limits.xbar.center == pytest.approx(6.0)
    assert limits.r.center == pytest.approx(8.0)


def test_the_xbar_limits_are_the_grand_mean_plus_or_minus_a2_times_rbar() -> None:
    limits = compute_limits(HAND_WORKED)
    a2 = constants_for(5).a2
    assert limits.xbar.upper == pytest.approx(6.0 + a2 * 8.0)
    assert limits.xbar.lower == pytest.approx(6.0 - a2 * 8.0)


def test_the_range_limits_are_d3_and_d4_times_rbar() -> None:
    limits = compute_limits(HAND_WORKED)
    c = constants_for(5)
    assert limits.r.upper == pytest.approx(c.d4_upper * 8.0)
    assert limits.r.lower == pytest.approx(c.d3_lower * 8.0)


def test_sigma_is_estimated_as_rbar_over_d2() -> None:
    limits = compute_limits(HAND_WORKED)
    assert limits.sigma_within == pytest.approx(8.0 / constants_for(5).d2)


def test_the_xbar_half_width_is_three_sigma_of_a_mean_not_of_a_part() -> None:
    """Means are sqrt(n) steadier than parts, so the limits are sqrt(n) tighter."""
    limits = compute_limits(HAND_WORKED)
    sigma_of_mean = limits.sigma_within / (5**0.5)
    assert limits.xbar.upper - limits.xbar.center == pytest.approx(
        3.0 * sigma_of_mean, rel=1e-6
    )


def test_the_lower_range_limit_is_floored_at_zero_for_five() -> None:
    limits = compute_limits(HAND_WORKED)
    assert limits.r.lower == 0.0
    assert limits.r.lower_is_floored


def test_a_larger_subgroup_gets_a_real_lower_range_limit() -> None:
    groups = [Subgroup(i, "X", tuple(float(v) for v in range(10))) for i in range(3)]
    limits = compute_limits(groups)
    assert limits.r.lower > 0.0
    assert not limits.r.lower_is_floored


# --------------------------------------------------------------------------
# Validation of the baseline
# --------------------------------------------------------------------------


def test_an_empty_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one baseline subgroup"):
        compute_limits([])


def test_mixed_subgroup_sizes_are_rejected() -> None:
    """One subgroup of three among fives would quietly bias R-bar downward."""
    mixed = [Subgroup(0, "X", (1.0, 2.0, 3.0)), Subgroup(1, "X", (1.0, 2.0))]
    with pytest.raises(ValueError, match="same size"):
        compute_limits(mixed)


def test_a_short_baseline_is_flagged_as_not_well_founded() -> None:
    assert not compute_limits(HAND_WORKED).is_well_founded


def test_a_full_baseline_is_well_founded() -> None:
    groups = bore_subgroups(FaultSchedule(), seed=1, count=25)
    assert compute_limits(groups).is_well_founded


def test_the_description_names_the_key_numbers() -> None:
    text = compute_limits(HAND_WORKED).describe()
    assert "Grand mean" in text
    assert "Sigma (within)" in text
    assert "A2" in text


# --------------------------------------------------------------------------
# Zones and distances
# --------------------------------------------------------------------------


def limits_at(center: float = 0.0, sigma: float = 1.0) -> ControlLimits:
    return ControlLimits(
        center=center, upper=center + 3 * sigma, lower=center - 3 * sigma, sigma=sigma
    )


def test_a_value_on_the_centre_line_is_zero_sigma_away() -> None:
    assert limits_at().sigma_distance(0.0) == pytest.approx(0.0)


def test_sigma_distance_is_signed() -> None:
    assert limits_at().sigma_distance(2.0) == pytest.approx(2.0)
    assert limits_at().sigma_distance(-2.0) == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.5, "+C"),
        (-0.5, "-C"),
        (1.5, "+B"),
        (-1.5, "-B"),
        (2.5, "+A"),
        (-2.5, "-A"),
        (3.5, "beyond"),
        (-3.5, "beyond"),
    ],
)
def test_zones_are_named_by_sigma_distance(value: float, expected: str) -> None:
    assert limits_at().zone(value) == expected


def test_a_value_inside_the_limits_is_not_flagged() -> None:
    assert not limits_at().is_beyond_limits(2.9)


def test_a_value_outside_the_limits_is_flagged() -> None:
    assert limits_at().is_beyond_limits(3.1)
    assert limits_at().is_beyond_limits(-3.1)


def test_the_width_is_the_distance_between_the_limits() -> None:
    assert limits_at(sigma=2.0).width == pytest.approx(12.0)


def test_a_zero_sigma_chart_reports_zero_distance() -> None:
    """A perfectly repeatable process must not divide by zero."""
    flat = ControlLimits(center=5.0, upper=5.0, lower=5.0, sigma=0.0)
    assert flat.sigma_distance(9.0) == 0.0


# --------------------------------------------------------------------------
# Plotting points
# --------------------------------------------------------------------------


def test_points_are_numbered_in_the_order_they_are_added() -> None:
    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))
    chart.add_many(bore_subgroups(FaultSchedule(), seed=2, count=6))
    assert [p.index for p in chart.points] == [0, 1, 2, 3, 4, 5]


def test_a_subgroup_of_the_wrong_size_cannot_be_plotted() -> None:
    chart = XbarRChart.fit(HAND_WORKED)
    with pytest.raises(ValueError, match="cannot be plotted"):
        chart.add(Subgroup(0, "X", (1.0, 2.0, 3.0)))


def test_evaluate_does_not_record_the_point() -> None:
    chart = XbarRChart.fit(HAND_WORKED)
    chart.evaluate(HAND_WORKED[0])
    assert chart.points == ()


def test_a_point_carries_its_subgroup_for_traceability() -> None:
    chart = XbarRChart.fit(HAND_WORKED)
    point = chart.add(HAND_WORKED[0])
    assert point.subgroup is HAND_WORKED[0]


def test_reset_clears_the_points_but_keeps_the_limits() -> None:
    chart = XbarRChart.fit(HAND_WORKED)
    before = chart.limits.xbar.upper
    chart.add_many(HAND_WORKED)
    chart.reset()
    assert chart.points == ()
    assert chart.limits.xbar.upper == before


def test_a_point_is_out_of_control_if_either_chart_flags_it() -> None:
    assert ChartPoint(0, 1.0, 1.0, 0.0, 0.0, True, False).out_of_control
    assert ChartPoint(0, 1.0, 1.0, 0.0, 0.0, False, True).out_of_control
    assert not ChartPoint(0, 1.0, 1.0, 0.0, 0.0, False, False).out_of_control


def test_the_point_table_has_a_row_per_point() -> None:
    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))
    chart.add_many(bore_subgroups(FaultSchedule(), seed=2, count=4))
    frame = chart.to_frame()
    assert len(frame) == 4
    assert "mean_sigma" in frame.columns


def test_an_empty_chart_still_produces_a_table_with_columns() -> None:
    chart = XbarRChart.fit(HAND_WORKED)
    assert list(chart.to_frame().columns)


# --------------------------------------------------------------------------
# The frozen-limits rule
# --------------------------------------------------------------------------


def test_the_limits_never_move_as_points_are_added() -> None:
    """Limits that chase the data can never detect a drift."""
    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))
    before = (chart.limits.xbar.upper, chart.limits.xbar.lower, chart.limits.r.upper)

    drift = FaultSchedule([ToolWear(tag="BoreDiameter", rate_per_hour=-0.08)], seed=1)
    chart.add_many(bore_subgroups(drift, seed=1, count=40))

    after = (chart.limits.xbar.upper, chart.limits.xbar.lower, chart.limits.r.upper)
    assert before == after


# --------------------------------------------------------------------------
# What each chart actually detects
# --------------------------------------------------------------------------


def test_a_healthy_process_signals_only_rarely() -> None:
    """False alarms must be rare, but they are not as rare as 1 in 370.

    Three sigma on a normal distribution is one point in 370, which is where
    that figure comes from. Two things push the real rate above it. The limits
    themselves were estimated from a finite baseline, so they carry error. And
    the distribution of a RANGE is right-skewed, so a symmetric three sigma
    upper limit is not as far into the tail as it looks.

    Measured across 800 healthy subgroups: about 0.5 percent on the X-bar chart
    and about 0.75 percent on the R chart. Both small, neither 0.27 percent.
    """
    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))
    for seed in range(50, 60):
        chart.add_many(bore_subgroups(FaultSchedule(), seed=seed, count=40))

    total = len(chart.points)
    assert total == 400
    assert len(chart.out_of_control_points) / total < 0.03


def test_a_mean_shift_trips_the_xbar_chart() -> None:
    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))
    shifted = FaultSchedule([MeanShift(tag="BoreDiameter", shift_sigma=2.0)], seed=1)
    chart.add_many(bore_subgroups(shifted, seed=7, count=20))
    assert any(p.mean_out_of_control for p in chart.points)


def test_a_mean_shift_leaves_the_range_chart_alone() -> None:
    """The parts are still consistent, they are just consistently wrong."""
    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))
    shifted = FaultSchedule([MeanShift(tag="BoreDiameter", shift_sigma=3.0)], seed=1)
    chart.add_many(bore_subgroups(shifted, seed=7, count=20))
    assert chart.range_is_in_control


def test_variance_inflation_trips_the_range_chart() -> None:
    """This is the failure the X-bar chart alone would miss."""
    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))
    noisy = FaultSchedule([VarianceInflation(tag="BoreDiameter", factor=3.0)], seed=1)
    chart.add_many(bore_subgroups(noisy, seed=7, count=20))
    assert not chart.range_is_in_control


def test_tool_wear_is_detected_before_any_part_goes_out_of_specification() -> None:
    """The headline claim of the whole project, asserted rather than hoped for."""
    config = load_config()
    spec = config.tag("BoreDiameter")
    n = config.subgroup_size

    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))

    wear = FaultSchedule([ToolWear(tag="BoreDiameter", rate_per_hour=-0.05)], seed=1)
    watched = bore_subgroups(wear, seed=1, count=40)
    chart.add_many(watched)

    signal = chart.first_signal
    assert signal is not None

    values = [v for group in watched for v in group.values]
    first_bad = next(
        (i for i, v in enumerate(values) if v < spec.lsl or v > spec.usl), None
    )
    assert first_bad is not None

    parts_at_signal = (signal.index + 1) * n
    assert parts_at_signal < first_bad + 1


def test_first_signal_is_none_when_nothing_goes_wrong() -> None:
    chart = XbarRChart.fit(bore_subgroups(FaultSchedule(), seed=1, count=25))
    chart.add_many(bore_subgroups(FaultSchedule(), seed=99, count=10))
    if chart.out_of_control_points:
        pytest.skip("this seed produced a false alarm, which is allowed")
    assert chart.first_signal is None


# --------------------------------------------------------------------------
# Control limits are not specification limits
# --------------------------------------------------------------------------


def test_the_control_limits_are_tighter_than_the_specification() -> None:
    """Subgroup means are less spread out than single parts, by sqrt(n).

    Drawing specification limits on an X-bar chart is a textbook error for
    exactly this reason: they are not comparable quantities.
    """
    spec = load_config().tag("BoreDiameter")
    limits = compute_limits(bore_subgroups(FaultSchedule(), seed=1, count=25))
    assert limits.xbar.lower > spec.lsl
    assert limits.xbar.upper < spec.usl


def test_the_control_limits_come_from_the_process_not_the_customer() -> None:
    """Doubling the process spread must widen the limits; the spec is unmoved."""
    tight = compute_limits(bore_subgroups(FaultSchedule(), seed=1, count=25))
    wide = compute_limits(
        bore_subgroups(
            FaultSchedule([VarianceInflation(tag="BoreDiameter", factor=2.0)], seed=1),
            seed=1,
            count=25,
        )
    )
    assert wide.xbar.width > 1.5 * tight.xbar.width
    assert wide.r.upper > 1.5 * tight.r.upper