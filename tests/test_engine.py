"""Tests for the SPC engine and the dashboard's chart builders."""

from __future__ import annotations

import pytest

from spc_opcua.config import load_config
from spc_opcua.dashboard import charts
from spc_opcua.simulator.faults import (
    FaultSchedule,
    MeanShift,
    ToolWear,
    VarianceInflation,
)
from spc_opcua.simulator.machine import MachineSimulator
from spc_opcua.spc.engine import SPCEngine
from spc_opcua.spc.nelson_rules import COMMON_RULES
from spc_opcua.spc.subgroups import Subgroup, subgroups_from_values


def bore_subgroups(faults: FaultSchedule, seed: int, count: int) -> list[Subgroup]:
    config = load_config()
    n = config.subgroup_size
    simulator = MachineSimulator(config, seed=seed, faults=faults)
    values: list[float] = []
    while len(values) < count * n:
        sample = simulator.step()
        if sample.part_completed:
            values.append(sample.values["BoreDiameter"])
    return subgroups_from_values(values, n, tag="BoreDiameter")


def healthy(seed: int, count: int) -> list[Subgroup]:
    return bore_subgroups(FaultSchedule(), seed, count)


def worn(seed: int, count: int, rate: float = -0.05) -> list[Subgroup]:
    return bore_subgroups(
        FaultSchedule([ToolWear(tag="BoreDiameter", rate_per_hour=rate)], seed=1),
        seed,
        count,
    )


def baselined(baseline: int = 25, window: int = 15, **kwargs) -> SPCEngine:
    """An engine that has already been through its baseline phase."""
    engine = SPCEngine(baseline_subgroups=baseline, capability_window=window, **kwargs)
    engine.add_many(healthy(seed=1, count=baseline))
    return engine


# --------------------------------------------------------------------------
# The two phases
# --------------------------------------------------------------------------


def test_the_engine_starts_in_the_baseline_phase() -> None:
    engine = SPCEngine(baseline_subgroups=25)
    assert engine.is_baselining
    assert engine.status == "BASELINING"
    assert engine.limits is None


def test_nothing_is_judged_during_the_baseline() -> None:
    """You cannot monitor against limits that do not exist yet."""
    engine = SPCEngine(baseline_subgroups=25)
    updates = engine.add_many(healthy(seed=1, count=25))
    assert all(u.phase == "baseline" for u in updates)
    assert all(u.point is None for u in updates)
    assert engine.alarms.alarms == ()


def test_the_limits_freeze_once_the_baseline_is_full() -> None:
    engine = baselined()
    assert not engine.is_baselining
    assert engine.limits is not None
    assert engine.limits.baseline_count == 25


def test_baseline_progress_climbs_to_one() -> None:
    engine = SPCEngine(baseline_subgroups=10)
    engine.add_many(healthy(seed=1, count=5))
    assert engine.baseline_progress == pytest.approx(0.5)
    engine.add_many(healthy(seed=2, count=5))
    assert engine.baseline_progress == 1.0


def test_a_baseline_below_two_subgroups_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 subgroups"):
        SPCEngine(baseline_subgroups=1)


def test_the_limits_do_not_move_during_monitoring() -> None:
    engine = baselined()
    before = engine.limits.xbar.upper
    engine.add_many(worn(seed=1, count=40))
    assert engine.limits.xbar.upper == before


# --------------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------------


def test_a_healthy_process_reports_in_control() -> None:
    engine = baselined()
    engine.add_many(healthy(seed=99, count=10))
    assert engine.status == "IN CONTROL"
    assert engine.alarms.active == ()


def test_tool_wear_drives_the_status_to_critical() -> None:
    engine = baselined()
    engine.add_many(worn(seed=1, count=40))
    assert engine.status == "CRITICAL"
    assert engine.alarms.active


def test_the_first_alarm_arrives_before_the_first_limit_break() -> None:
    """A pattern rule fires while every point is still inside the limits."""
    engine = baselined(rules=COMMON_RULES)
    updates = engine.add_many(worn(seed=1, count=40))

    first_alarm = next(u for u in updates if u.raised)
    first_break = next(u for u in updates if u.point.mean_out_of_control)
    assert first_alarm.point.index < first_break.point.index


def test_a_mean_shift_alarms_on_the_xbar_chart_only() -> None:
    engine = baselined()
    shifted = bore_subgroups(
        FaultSchedule([MeanShift(tag="BoreDiameter", shift_sigma=3.0)], seed=1),
        seed=7,
        count=20,
    )
    engine.add_many(shifted)
    charts_alarmed = {alarm.chart for alarm in engine.alarms.alarms}
    assert charts_alarmed == {"X-bar"}


def test_variance_inflation_alarms_on_the_r_chart() -> None:
    engine = baselined()
    noisy = bore_subgroups(
        FaultSchedule([VarianceInflation(tag="BoreDiameter", factor=3.0)], seed=1),
        seed=7,
        count=20,
    )
    engine.add_many(noisy)
    assert any(alarm.chart == "R" for alarm in engine.alarms.alarms)


def test_the_engine_counts_what_it_has_monitored() -> None:
    engine = baselined()
    engine.add_many(healthy(seed=3, count=12))
    assert engine.subgroups_monitored == 12
    assert len(engine.points) == 12


def test_only_the_chosen_rules_are_applied() -> None:
    engine = baselined(rules=[1])
    engine.add_many(worn(seed=1, count=40))
    assert engine.alarms.rules_seen() == {1}


# --------------------------------------------------------------------------
# Capability
# --------------------------------------------------------------------------


def test_capability_waits_for_a_full_window() -> None:
    engine = baselined(window=15)
    engine.add_many(healthy(seed=3, count=14))
    assert engine.capability is None
    engine.add_many(healthy(seed=4, count=1))
    assert engine.capability is not None


def test_the_capability_trend_grows_one_point_at_a_time() -> None:
    engine = baselined(window=10)
    engine.add_many(healthy(seed=3, count=25))
    assert len(engine.capability_trend) == 25 - 10 + 1


def test_capability_falls_as_a_tool_wears() -> None:
    engine = baselined(window=15)
    engine.add_many(worn(seed=1, count=45))
    trend = engine.capability_trend
    assert trend[-1].cpk < trend[0].cpk - 0.3


# --------------------------------------------------------------------------
# Output shapes the dashboard depends on
# --------------------------------------------------------------------------


def test_the_chart_table_has_a_row_per_subgroup() -> None:
    engine = baselined()
    engine.add_many(healthy(seed=3, count=8))
    frame = engine.chart_frame()
    assert len(frame) == 8
    for column in ("subgroup", "mean", "range", "mean_ooc", "range_ooc"):
        assert column in frame.columns


def test_the_tables_have_columns_even_when_empty() -> None:
    """The dashboard draws before any data arrives and must not crash."""
    engine = SPCEngine(baseline_subgroups=25)
    assert list(engine.chart_frame().columns)
    assert list(engine.capability_frame().columns)
    assert list(engine.alarm_frame().columns)


def test_the_capability_table_carries_its_subgroup_position() -> None:
    engine = baselined(window=10)
    engine.add_many(healthy(seed=3, count=15))
    frame = engine.capability_frame()
    assert frame["subgroup"].iloc[0] == 9


def test_the_alarm_table_can_hide_cleared_alarms() -> None:
    engine = baselined()
    engine.add_many(worn(seed=1, count=40))
    assert len(engine.alarm_frame(include_cleared=False)) <= len(engine.alarm_frame())


def test_the_summary_reports_the_phase_and_the_limits() -> None:
    engine = SPCEngine(baseline_subgroups=25)
    assert "baseline" in engine.summary()
    engine.add_many(healthy(seed=1, count=25))
    engine.add_many(healthy(seed=2, count=20))
    text = engine.summary()
    assert "Grand mean" in text
    assert "Cpk" in text


# --------------------------------------------------------------------------
# The Plotly figures
# --------------------------------------------------------------------------


def test_every_chart_builds_from_engine_output() -> None:
    engine = baselined(window=10)
    engine.add_many(worn(seed=1, count=25))
    spec = load_config().tag("BoreDiameter")
    rows = [point.as_row() for point in engine.points]
    capability_rows = engine.capability_frame().to_dict("records")

    assert charts.xbar_chart(rows, engine.limits, spec).data
    assert charts.r_chart(rows, engine.limits, spec).data
    assert charts.cpk_chart(capability_rows).data
    assert charts.individuals_chart([20.0, 19.99, 20.01], spec).data


def test_out_of_control_points_get_their_own_trace() -> None:
    """A second trace means a different colour AND a different symbol."""
    engine = baselined()
    engine.add_many(worn(seed=1, count=40))
    spec = load_config().tag("BoreDiameter")
    rows = [point.as_row() for point in engine.points]
    figure = charts.xbar_chart(rows, engine.limits, spec)
    assert len(figure.data) == 2
    assert figure.data[1].marker.symbol == "x-thin"


def test_a_clean_chart_has_only_the_data_trace() -> None:
    engine = baselined()
    engine.add_many(healthy(seed=99, count=6))
    spec = load_config().tag("BoreDiameter")
    rows = [point.as_row() for point in engine.points]
    assert len(charts.xbar_chart(rows, engine.limits, spec).data) == 1


def test_specification_limits_never_appear_on_the_control_charts() -> None:
    """A part tolerance across subgroup means would be misleading."""
    engine = baselined()
    engine.add_many(healthy(seed=3, count=6))
    spec = load_config().tag("BoreDiameter")
    rows = [point.as_row() for point in engine.points]

    for figure in (
        charts.xbar_chart(rows, engine.limits, spec),
        charts.r_chart(rows, engine.limits, spec),
    ):
        labels = {shape.text for shape in figure.layout.annotations}
        assert "USL" not in labels
        assert "LSL" not in labels


def test_specification_limits_do_appear_on_the_individuals_chart() -> None:
    spec = load_config().tag("BoreDiameter")
    figure = charts.individuals_chart([20.0, 19.99, 20.01], spec)
    labels = {shape.text for shape in figure.layout.annotations}
    assert {"USL", "LSL", "nominal"} <= labels


def test_the_empty_figures_carry_a_message() -> None:
    spec = load_config().tag("BoreDiameter")
    assert charts.individuals_chart([], spec).layout.annotations
    assert charts.cpk_chart([], window=15, monitored=3).layout.annotations[0].text


def test_every_status_word_has_a_colour() -> None:
    for word in ("IN CONTROL", "BASELINING", "WARNING", "CRITICAL", "DISCONNECTED"):
        assert word in charts.STATUS_COLOURS