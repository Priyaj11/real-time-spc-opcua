"""Tests for the eight Nelson Rules.

Each rule gets its own section: one series built to trip it, one to prove it
does not trip a point too early, and boundary cases around the thresholds.
Everything here works on plain lists of sigma distances, so no chart, machine
or network is involved.
"""

from __future__ import annotations

import pytest

from spc_opcua.simulator.faults import FaultSchedule, ToolWear
from spc_opcua.spc.nelson_rules import (
    ALL_RULES,
    COMMON_RULES,
    MAX_RULE_WINDOW,
    RULES,
    TRIGGERING_DATA,
    NelsonMonitor,
    Violation,
    apply_rules,
    rule_1,
    rule_2,
    rule_3,
    rule_4,
    rule_5,
    rule_6,
    rule_7,
    rule_8,
    _bore_sigmas,
)

# A deliberately unremarkable series: no long one-sided run, no monotone
# stretch, no clean alternation, and a few points past one sigma so it cannot
# be accused of hugging the centre line either.
QUIET = [
    0.4, -0.9, 1.3, 0.2, -0.5, 0.8, -1.2, 0.1,
    0.6, -0.3, -0.7, 1.1, 0.5, -0.2, 0.9, -1.0,
    0.3, 0.7, -0.6, 0.2, 1.2, -0.4, 0.1, -0.8,
]


def fired(violations: list[Violation]) -> set[int]:
    return {v.rule for v in violations}


# --------------------------------------------------------------------------
# Every rule fires on its own documented data, and only its own
# --------------------------------------------------------------------------


@pytest.mark.parametrize("number", ALL_RULES)
def test_each_rule_fires_on_its_documented_data(number: int) -> None:
    assert RULES[number].check(TRIGGERING_DATA[number])


@pytest.mark.parametrize("number", ALL_RULES)
def test_each_documented_series_trips_only_its_own_rule(number: int) -> None:
    """Kept isolated so the docstring examples teach one idea at a time."""
    assert fired(apply_rules(TRIGGERING_DATA[number])) == {number}


@pytest.mark.parametrize("number", ALL_RULES)
def test_no_rule_fires_on_a_quiet_healthy_series(number: int) -> None:
    assert RULES[number].check(QUIET) == []


@pytest.mark.parametrize("number", ALL_RULES)
def test_every_rule_is_registered_with_a_description(number: int) -> None:
    spec = RULES[number]
    assert spec.number == number
    assert spec.name and spec.detects and spec.matters
    assert 1 <= spec.window <= MAX_RULE_WINDOW


# --------------------------------------------------------------------------
# Rule 1: one point beyond three sigma
# --------------------------------------------------------------------------


def test_rule_1_fires_above_and_below() -> None:
    assert len(rule_1([3.5, 0.0, -3.5])) == 2


def test_rule_1_does_not_fire_exactly_on_the_limit() -> None:
    """A point sitting on the limit is inside it. Only beyond counts."""
    assert rule_1([3.0, -3.0]) == []
    assert rule_1([3.0001]) != []


def test_rule_1_reports_the_single_offending_point() -> None:
    violation = rule_1([0.0, 0.0, 4.2])[0]
    assert violation.end_index == 2
    assert violation.indices == (2,)
    assert violation.span == 1
    assert "+4.20" in violation.detail


# --------------------------------------------------------------------------
# Rule 2: nine points in a row on one side
# --------------------------------------------------------------------------


def test_rule_2_needs_nine_not_eight() -> None:
    assert rule_2([0.5] * 8) == []
    assert rule_2([0.5] * 9) != []


def test_rule_2_fires_below_the_centre_too() -> None:
    violation = rule_2([-0.2] * 9)[0]
    assert "below" in violation.detail


def test_a_point_exactly_on_the_centre_line_breaks_the_run() -> None:
    """Zero is on neither side, so it cannot extend a one-sided run."""
    assert rule_2([0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5]) == []


def test_a_crossing_breaks_the_run() -> None:
    assert rule_2([0.5] * 4 + [-0.5] + [0.5] * 4) == []


def test_rule_2_reports_the_nine_points_involved() -> None:
    violation = rule_2([0.0] + [0.5] * 9)[0]
    assert violation.end_index == 9
    assert violation.indices == tuple(range(1, 10))
    assert violation.span == 9


# --------------------------------------------------------------------------
# Rule 3: six in a row rising or falling
# --------------------------------------------------------------------------


def test_rule_3_needs_six_not_five() -> None:
    assert rule_3([0.1, 0.2, 0.3, 0.4, 0.5]) == []
    assert rule_3([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) != []


def test_rule_3_catches_a_falling_trend() -> None:
    violation = rule_3([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])[0]
    assert "falling" in violation.detail


def test_two_equal_values_break_a_trend() -> None:
    """A flat pair is not a trend, and a plateau should not count as one."""
    assert rule_3([0.1, 0.2, 0.2, 0.3, 0.4, 0.5]) == []


def test_rule_3_fires_well_inside_the_control_limits() -> None:
    """The whole point: a trend is visible long before anything hits a limit."""
    gentle = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    assert all(abs(v) < 1.0 for v in gentle)
    assert rule_3(gentle) != []


# --------------------------------------------------------------------------
# Rule 4: fourteen alternating
# --------------------------------------------------------------------------


def test_rule_4_needs_fourteen_not_thirteen() -> None:
    assert rule_4([0.0, 1.0] * 6 + [0.0]) == []  # 13 points
    assert rule_4([0.0, 1.0] * 7) != []  # 14 points


def test_a_repeat_breaks_the_alternation() -> None:
    series = [0.0, 1.0] * 7
    series[6] = series[5]  # two equal values in the middle
    assert rule_4(series) == []


def test_two_steps_in_the_same_direction_break_the_alternation() -> None:
    series = [0.0, 1.0] * 7
    series[2] = 2.0  # now it goes up, up rather than up, down
    assert rule_4(series) == []


# --------------------------------------------------------------------------
# Rule 5: two of three beyond two sigma, same side
# --------------------------------------------------------------------------


def test_rule_5_fires_on_two_of_three() -> None:
    assert rule_5([2.5, 0.1, 2.4]) != []


def test_rule_5_needs_the_two_points_on_the_same_side() -> None:
    """One high and one low is a spread problem, which the R chart catches."""
    assert rule_5([2.5, 0.1, -2.4]) == []


def test_rule_5_does_not_fire_exactly_at_two_sigma() -> None:
    assert rule_5([2.0, 0.0, 2.0]) == []
    assert rule_5([2.01, 0.0, 2.01]) != []


def test_rule_5_needs_the_newest_point_to_be_one_of_the_extremes() -> None:
    """Otherwise the same window would re-report as it slides past."""
    assert rule_5([2.5, 2.4, 0.1]) == []


def test_rule_5_fires_before_anything_leaves_the_limits() -> None:
    series = [2.5, 0.1, 2.4]
    assert all(abs(v) < 3.0 for v in series)
    assert rule_5(series) != []


# --------------------------------------------------------------------------
# Rule 6: four of five beyond one sigma, same side
# --------------------------------------------------------------------------


def test_rule_6_fires_on_four_of_five() -> None:
    assert rule_6([1.2, 1.5, 0.2, 1.1, 1.3]) != []


def test_three_of_five_is_not_enough() -> None:
    assert rule_6([1.2, 1.5, 0.2, 0.1, 1.3]) == []


def test_rule_6_needs_them_on_the_same_side() -> None:
    assert rule_6([1.2, -1.5, 1.1, -1.3, 1.4]) == []


def test_rule_6_does_not_fire_exactly_at_one_sigma() -> None:
    assert rule_6([1.0, 1.0, 0.0, 1.0, 1.0]) == []


# --------------------------------------------------------------------------
# Rule 7: fifteen hugging the centre
# --------------------------------------------------------------------------


def test_rule_7_needs_fifteen_not_fourteen() -> None:
    calm = [0.2, 0.1, -0.1, -0.2, 0.3]
    assert rule_7((calm * 3)[:14]) == []
    assert rule_7(calm * 3) != []


def test_one_point_outside_one_sigma_breaks_the_hug() -> None:
    series = (([0.2, 0.1, -0.1, -0.2, 0.3]) * 3)[:]
    series[7] = 1.5
    assert rule_7(series) == []


def test_a_point_exactly_at_one_sigma_breaks_the_hug() -> None:
    series = ([0.2, 0.1, -0.1, -0.2, 0.3] * 3)[:]
    series[7] = 1.0
    assert rule_7(series) == []


def test_rule_7_catches_a_stuck_sensor() -> None:
    """A frozen reading sits exactly on wherever it froze, forever."""
    assert rule_7([0.05] * 15) != []


def test_rule_7_warns_that_the_limits_may_be_wrong() -> None:
    violation = rule_7([0.2, 0.1, -0.1, -0.2, 0.3] * 3)[0]
    assert "too wide" in violation.detail


# --------------------------------------------------------------------------
# Rule 8: eight avoiding the centre, both sides
# --------------------------------------------------------------------------


def test_rule_8_fires_on_a_two_stream_pattern() -> None:
    assert rule_8([1.5, -1.5, 1.6, -1.4, 1.5, -1.6, 1.4, -1.5]) != []


def test_rule_8_needs_eight_not_seven() -> None:
    series = [1.5, -1.5, 1.6, -1.4, 1.5, -1.6, 1.4, -1.5]
    assert rule_8(series[:7]) == []
    assert rule_8(series) != []


def test_rule_8_needs_points_on_both_sides() -> None:
    """A one-sided run beyond one sigma is rule 6's job, not rule 8's."""
    assert rule_8([1.5] * 8) == []


def test_a_point_inside_one_sigma_breaks_the_avoidance() -> None:
    series = [1.5, -1.5, 1.6, -1.4, 1.5, -1.6, 1.4, -1.5]
    series[4] = 0.2
    assert rule_8(series) == []


def test_rule_8_suggests_a_mixture() -> None:
    violation = rule_8([1.5, -1.5, 1.6, -1.4, 1.5, -1.6, 1.4, -1.5])[0]
    assert "mixture" in violation.detail


# --------------------------------------------------------------------------
# Applying several rules at once
# --------------------------------------------------------------------------


def test_apply_rules_runs_every_rule_by_default() -> None:
    series = [0.0, 0.0, 3.5] + [0.5] * 9
    assert fired(apply_rules(series)) == {1, 2}


def test_apply_rules_can_be_narrowed() -> None:
    series = [0.0, 0.0, 3.5] + [0.5] * 9
    assert fired(apply_rules(series, rules=[1])) == {1}


def test_violations_come_back_in_the_order_they_completed() -> None:
    series = [0.5] * 9 + [0.0, 0.0, 4.0]
    ends = [v.end_index for v in apply_rules(series)]
    assert ends == sorted(ends)


def test_an_unknown_rule_number_is_rejected() -> None:
    with pytest.raises(KeyError, match="There is no Nelson Rule 9"):
        apply_rules([0.0], rules=[9])


def test_the_common_set_is_a_subset_of_all_eight() -> None:
    assert set(COMMON_RULES) < set(ALL_RULES)


def test_an_empty_series_produces_nothing() -> None:
    assert apply_rules([]) == []


def test_a_single_point_can_still_trip_rule_1() -> None:
    assert fired(apply_rules([5.0])) == {1}


# --------------------------------------------------------------------------
# The streaming monitor
# --------------------------------------------------------------------------


def test_the_monitor_reports_a_violation_at_the_point_it_completes() -> None:
    monitor = NelsonMonitor(rules=[1])
    assert monitor.add(0.5) == []
    fired_now = monitor.add(4.0)
    assert len(fired_now) == 1
    assert fired_now[0].end_index == 1


def test_the_monitor_agrees_with_scanning_the_whole_series() -> None:
    """Streaming and batch must find exactly the same patterns."""
    series = [0.5] * 9 + [-0.2, -0.4, 4.0, 2.5, 0.1, 2.4] + [0.1, 0.2, 0.3, 0.4, 0.5]
    monitor = NelsonMonitor()
    monitor.add_many(series)
    streamed = {(v.rule, v.end_index) for v in monitor.violations}
    batched = {(v.rule, v.end_index) for v in apply_rules(series)}
    assert streamed == batched


def test_the_monitor_counts_the_points_it_has_seen() -> None:
    monitor = NelsonMonitor()
    monitor.add_many([0.1] * 20)
    assert monitor.points_seen == 20


def test_indices_stay_absolute_beyond_the_window() -> None:
    """The monitor keeps only 15 points, but indices must not restart."""
    monitor = NelsonMonitor(rules=[1])
    monitor.add_many([0.1] * 40)
    violation = monitor.add(5.0)[0]
    assert violation.end_index == 40


def test_the_monitor_reports_which_rules_have_fired() -> None:
    monitor = NelsonMonitor()
    monitor.add_many([0.5] * 9 + [4.0])
    assert monitor.rules_fired() == {1, 2}


def test_the_first_violation_is_the_earliest_one() -> None:
    monitor = NelsonMonitor()
    monitor.add_many([0.5] * 9 + [4.0])
    assert monitor.first_violation.rule == 2


def test_there_is_no_first_violation_on_a_quiet_process() -> None:
    monitor = NelsonMonitor()
    monitor.add_many(QUIET)
    assert monitor.first_violation is None
    assert monitor.violations == ()


def test_reset_clears_points_and_violations() -> None:
    monitor = NelsonMonitor()
    monitor.add_many([0.5] * 9 + [4.0])
    monitor.reset()
    assert monitor.points_seen == 0
    assert monitor.violations == ()


def test_the_monitor_rejects_an_unknown_rule_number() -> None:
    with pytest.raises(KeyError, match="There is no Nelson Rule 0"):
        NelsonMonitor(rules=[0])


def test_the_monitor_only_needs_fifteen_points_of_memory() -> None:
    """No rule looks further back, so memory must not grow with the run."""
    assert max(spec.window for spec in RULES.values()) == MAX_RULE_WINDOW


# --------------------------------------------------------------------------
# The trade-off, measured on real simulated data
# --------------------------------------------------------------------------


def tool_wear_sigmas() -> list[float]:
    wear = FaultSchedule(
        [ToolWear(tag="BoreDiameter", start_s=0.0, rate_per_hour=-0.05)], seed=1
    )
    return _bore_sigmas(wear, seed=1, count=40)


def healthy_sigmas(seed: int) -> list[float]:
    return _bore_sigmas(FaultSchedule(), seed=seed, count=40)


def test_more_rules_detect_tool_wear_no_later_than_rule_one_alone() -> None:
    sigmas = tool_wear_sigmas()

    def first_at(rules) -> int:
        monitor = NelsonMonitor(rules=rules)
        monitor.add_many(sigmas)
        assert monitor.first_violation is not None
        return monitor.first_violation.end_index

    assert first_at(ALL_RULES) <= first_at(COMMON_RULES) <= first_at((1,))


def test_the_pattern_rules_catch_tool_wear_strictly_earlier_here() -> None:
    """Measured: rule 1 alone fires at subgroup 14, the common five at 11."""
    sigmas = tool_wear_sigmas()

    rule_one = NelsonMonitor(rules=[1])
    rule_one.add_many(sigmas)

    common = NelsonMonitor(rules=COMMON_RULES)
    common.add_many(sigmas)

    assert common.first_violation.end_index < rule_one.first_violation.end_index


def test_more_rules_also_mean_more_false_alarms() -> None:
    """The honest cost. Measured at roughly 0.5 percent against 5 percent."""

    def flagged_points(rules) -> int:
        total = 0
        for seed in range(50, 60):
            monitor = NelsonMonitor(rules=rules)
            monitor.add_many(healthy_sigmas(seed))
            total += len({v.end_index for v in monitor.violations})
        return total

    one = flagged_points((1,))
    common = flagged_points(COMMON_RULES)
    everything = flagged_points(ALL_RULES)

    assert one < common <= everything
    assert common > 3 * max(one, 1)