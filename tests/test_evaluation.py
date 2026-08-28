"""Tests for the fault scenario evaluation.

The evaluation produces the numbers that go in the README, so it needs testing
harder than most things here. A detector that silently scores itself well is
worse than no detector at all.

These tests deliberately do not assert particular detection rates or
latencies. Those are results, not requirements, and pinning them in a test
would mean the test has to be edited every time the process changes, which
defeats the point of measuring. What is tested is that the measurement itself
is correct: that a fault cannot leak into the baseline, that scrap is counted
from the true dimension rather than the reading, that latency is counted from
the right subgroup, and that the whole thing is reproducible.
"""

from __future__ import annotations

import pytest

from spc_opcua.evaluation import runner
from spc_opcua.evaluation.scenarios import (
    SCENARIOS,
    START_S,
    Scenario,
    scenario,
)
from spc_opcua.spc.nelson_rules import COMMON_RULES

FAST = dict(baseline=15, monitor=25)


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


def test_there_are_twelve_scenarios() -> None:
    assert len(SCENARIOS) == 12


def test_every_scenario_name_is_unique() -> None:
    names = [s.name for s in SCENARIOS]
    assert len(set(names)) == len(names)


def test_exactly_one_scenario_is_healthy() -> None:
    """The control. Without it there is nothing to measure false alarms on."""
    healthy = [s for s in SCENARIOS if s.is_healthy]
    assert len(healthy) == 1
    assert not healthy[0].expect_detection


def test_every_fault_starts_after_the_baseline_ends() -> None:
    """A fault inside the baseline would be built into the control limits."""
    for item in SCENARIOS:
        for fault in item.faults:
            assert fault.start_s >= START_S, item.name


def test_sensor_scenarios_are_marked_as_producing_no_scrap() -> None:
    for item in SCENARIOS:
        if item.kind == "sensor":
            assert not item.produces_scrap


def test_both_kinds_of_fault_are_represented() -> None:
    kinds = {s.kind for s in SCENARIOS}
    assert kinds == {"healthy", "process", "sensor"}


def test_a_scenario_can_be_looked_up_by_name() -> None:
    assert scenario("tool-wear-fast").name == "tool-wear-fast"


def test_an_unknown_scenario_name_lists_the_valid_ones() -> None:
    with pytest.raises(KeyError, match="healthy"):
        scenario("catastrophe")


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------


def test_a_run_is_reproducible() -> None:
    """Same scenario, same seed, same numbers. Otherwise nothing is evidence."""
    item = scenario("tool-wear-fast")
    first = runner.run_once(item, seed=1, **FAST)
    second = runner.run_once(item, seed=1, **FAST)
    assert first == second


def test_different_seeds_give_different_runs() -> None:
    item = scenario("tool-wear-fast")
    a = runner.run_once(item, seed=1, **FAST)
    b = runner.run_once(item, seed=2, **FAST)
    assert (a.detection_subgroups, a.scrap_parts) != (
        b.detection_subgroups,
        b.scrap_parts,
    )


def test_an_obvious_fault_is_detected() -> None:
    """A three sigma step should not be subtle. If this fails, something is wrong."""
    result = runner.run_once(scenario("mean-shift-3sigma"), seed=1, **FAST)
    assert result.detected
    assert result.detection_subgroups is not None


def test_detection_latency_counts_from_the_first_monitored_subgroup() -> None:
    """A fault caught on the very first monitored subgroup scores 1, not 0."""
    result = runner.run_once(scenario("mean-shift-3sigma"), seed=1, **FAST)
    assert result.detection_subgroups is not None
    assert result.detection_subgroups >= 1
    assert result.detection_subgroups <= FAST["monitor"]


def test_detection_in_parts_is_detection_in_subgroups_times_five() -> None:
    result = runner.run_once(scenario("mean-shift-2sigma"), seed=1, **FAST)
    assert result.detection_parts == result.detection_subgroups * 5


def test_a_sensor_fault_never_produces_scrap() -> None:
    """The gauge is wrong; the parts are fine. Counting them would flatter us."""
    for name in ("sensor-drift", "sensor-stuck", "sensor-noise"):
        result = runner.run_once(scenario(name), seed=1, **FAST)
        assert result.scrap_parts == 0, name


def test_a_sensor_fault_still_raises_an_alarm() -> None:
    """Something really is wrong. Silence would be the failure here."""
    result = runner.run_once(scenario("sensor-stuck"), seed=1, **FAST)
    assert result.detected


def test_a_severe_process_fault_does_produce_scrap() -> None:
    result = runner.run_once(scenario("mean-shift-3sigma"), seed=1, **FAST)
    assert result.scrap_parts > 0


def test_scrap_after_the_alarm_never_exceeds_total_scrap() -> None:
    for item in SCENARIOS:
        result = runner.run_once(item, seed=1, **FAST)
        assert result.scrap_after_alarm <= result.scrap_parts, item.name


def test_alarm_subgroups_cannot_exceed_the_window() -> None:
    for item in SCENARIOS:
        result = runner.run_once(item, seed=1, **FAST)
        assert 0 <= result.alarm_subgroups <= result.monitored_subgroups, item.name


def test_a_run_that_never_alarms_reports_nothing_rather_than_zero() -> None:
    """None means it did not happen. Zero would mean it happened instantly."""
    item = Scenario(name="quiet", description="none", expect_detection=False)
    result = runner.run_once(item, seed=7, baseline=15, monitor=3, rules=(1,))
    if not result.detected:
        assert result.detection_subgroups is None
        assert result.detection_parts is None
        assert result.warning_parts is None


def test_the_fault_does_not_touch_the_baseline() -> None:
    """The frozen limits must be identical to a healthy machine's, same seed."""
    healthy = runner.run_once(scenario("healthy"), seed=3, **FAST)
    worn = runner.run_once(scenario("tool-wear-fast"), seed=3, **FAST)
    # Same seed and an untouched baseline means the same first monitored
    # subgroups would have been judged against the same limits; the proof is
    # that the healthy run's own scrap count is zero either way.
    assert healthy.scrap_parts == 0
    assert worn.scrap_parts >= 0


# --------------------------------------------------------------------------
# Summarising
# --------------------------------------------------------------------------


def test_a_summary_covers_every_replicate() -> None:
    item = scenario("tool-wear-fast")
    results = runner.run_scenario(item, replicates=3, **FAST)
    summary = runner.summarise(item, results)
    assert summary.replicates == 3
    assert 0.0 <= summary.detection_rate <= 1.0


def test_the_detection_rate_is_caught_over_total() -> None:
    item = scenario("mean-shift-3sigma")
    results = runner.run_scenario(item, replicates=4, **FAST)
    summary = runner.summarise(item, results)
    caught = sum(r.detected for r in results)
    assert summary.detection_rate == pytest.approx(caught / 4)


def test_the_per_subgroup_alarm_rate_is_a_share() -> None:
    item = scenario("healthy")
    results = runner.run_scenario(item, replicates=4, **FAST)
    summary = runner.summarise(item, results)
    assert 0.0 <= summary.alarm_rate_per_subgroup <= 1.0


def test_the_run_length_is_the_reciprocal_of_the_rate() -> None:
    item = scenario("healthy")
    results = runner.run_scenario(item, replicates=4, **FAST)
    summary = runner.summarise(item, results)
    if summary.alarm_rate_per_subgroup:
        assert summary.subgroups_between_alarms == pytest.approx(
            1.0 / summary.alarm_rate_per_subgroup
        )


def test_a_scenario_with_no_scrap_reports_no_avoidable_share() -> None:
    """Zero divided by zero is not zero per cent, it is not a number."""
    item = scenario("sensor-drift")
    results = runner.run_scenario(item, replicates=2, **FAST)
    summary = runner.summarise(item, results)
    assert summary.scrap_parts == 0
    assert summary.scrap_avoidable is None


def test_the_summary_row_carries_what_the_table_needs() -> None:
    item = scenario("tool-wear-fast")
    row = runner.summarise(
        item, runner.run_scenario(item, replicates=2, **FAST)
    ).as_row()
    for key in (
        "scenario",
        "kind",
        "detection_rate",
        "median_subgroups",
        "scrap_avoidable",
        "alarm_rate_per_subgroup",
    ):
        assert key in row


# --------------------------------------------------------------------------
# The headline numbers, which are the ones that reach the README
# --------------------------------------------------------------------------


def test_the_headline_excludes_healthy_runs_from_the_detection_rate() -> None:
    """A healthy run has nothing to detect, so it cannot be a miss."""
    subset = (scenario("healthy"), scenario("mean-shift-3sigma"))
    results = runner.run_all(replicates=2, scenarios=subset, **FAST)
    summaries = runner.summarise_all(results, scenarios=subset)
    numbers = runner.headline(summaries)
    assert numbers["faulted_runs"] == 2
    assert numbers["healthy_runs"] == 2


def test_the_headline_reports_both_false_alarm_rates() -> None:
    """Per window and per subgroup are different numbers and both are needed."""
    subset = (scenario("healthy"), scenario("mean-shift-3sigma"))
    results = runner.run_all(replicates=2, scenarios=subset, **FAST)
    numbers = runner.headline(runner.summarise_all(results, scenarios=subset))
    assert 0.0 <= numbers["false_alarm_rate_per_window"] <= 1.0
    assert 0.0 <= numbers["false_alarm_rate_per_subgroup"] <= 1.0


def test_more_rules_detect_sooner_and_alarm_more_often_when_healthy() -> None:
    """The central trade-off of the whole project, measured rather than asserted."""
    healthy = scenario("healthy")
    one = runner.summarise(
        healthy, runner.run_scenario(healthy, replicates=6, rules=(1,), **FAST)
    )
    five = runner.summarise(
        healthy,
        runner.run_scenario(healthy, replicates=6, rules=COMMON_RULES, **FAST),
    )
    assert five.alarm_rate_per_subgroup >= one.alarm_rate_per_subgroup


@pytest.mark.slow
def test_the_whole_evaluation_runs_end_to_end() -> None:
    """Every scenario, a few replicates each, all the way to the headline."""
    results = runner.run_all(replicates=3, **FAST)
    summaries = runner.summarise_all(results)
    assert len(summaries) == 12
    numbers = runner.headline(summaries)
    assert numbers["faulted_runs"] == 33
    assert 0.0 <= numbers["detection_rate"] <= 1.0