"""Tests for fault injection."""

from __future__ import annotations

import numpy as np
import pytest

from spc_opcua.config import MachineConfig, load_config
from spc_opcua.simulator.faults import (
    Fault,
    FaultSchedule,
    MeanShift,
    Outlier,
    ProcessEffect,
    SensorDrift,
    SensorEffect,
    SensorNoise,
    SensorStuck,
    ToolWear,
    VarianceInflation,
)
from spc_opcua.simulator.machine import MachineSimulator, Sample

HOUR = 3600.0


@pytest.fixture
def config() -> MachineConfig:
    return load_config()


def published(samples: list[Sample], tag: str) -> list[float]:
    """What the machine reported for one tag."""
    return [s.values[tag] for s in samples]


def actual(samples: list[Sample], tag: str) -> list[float]:
    """What was physically true for one tag."""
    return [s.truth[tag] for s in samples]


def bores(samples: list[Sample], truth: bool = True) -> list[float]:
    """One bore reading per finished part."""
    source = "truth" if truth else "values"
    return [getattr(s, source)["BoreDiameter"] for s in samples if s.part_completed]


def run(config: MachineConfig, seconds: float, *faults: Fault) -> list[Sample]:
    """Run a seeded simulation with the given faults attached."""
    schedule = FaultSchedule(faults, seed=42)
    return MachineSimulator(config, seed=42, faults=schedule).run_seconds(seconds)


# --------------------------------------------------------------------------
# The neutrality guarantee: faults must not disturb the healthy stream
# --------------------------------------------------------------------------


def test_empty_schedule_matches_no_schedule_exactly(config: MachineConfig) -> None:
    plain = MachineSimulator(config, seed=42).run(2000)
    empty = MachineSimulator(config, seed=42, faults=FaultSchedule()).run(2000)
    assert published(plain, "Torque") == published(empty, "Torque")
    assert bores(plain) == bores(empty)


def test_a_fault_that_has_not_started_changes_nothing(config: MachineConfig) -> None:
    """Attaching a future fault must not shift a single value beforehand."""
    plain = MachineSimulator(config, seed=42).run_seconds(300.0)
    pending = run(
        config,
        300.0,
        ToolWear(tag="BoreDiameter", start_s=100_000.0, rate_per_hour=-0.05),
        Outlier(tag="Torque", start_s=100_000.0, probability=1.0, magnitude_sigma=8.0),
    )
    for tag in ("BoreDiameter", "Torque", "Temperature", "Vibration"):
        assert published(plain, tag) == published(pending, tag)


def test_a_fault_on_another_tag_leaves_this_tag_alone(config: MachineConfig) -> None:
    plain = MachineSimulator(config, seed=42).run_seconds(600.0)
    faulted = run(config, 600.0, MeanShift(tag="Torque", shift_sigma=5.0))
    assert bores(plain) == bores(faulted)
    assert published(plain, "Torque") != published(faulted, "Torque")


def test_truth_and_values_agree_when_only_process_faults_run(
    config: MachineConfig,
) -> None:
    samples = run(config, 600.0, ToolWear(tag="BoreDiameter", rate_per_hour=-0.05))
    for s in samples:
        assert s.values == s.truth


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_the_same_faults_and_seed_reproduce_exactly(config: MachineConfig) -> None:
    spec = dict(tag="BoreDiameter", start_s=60.0, probability=0.2, magnitude_sigma=5.0)
    first = run(config, 900.0, Outlier(**spec))
    second = run(config, 900.0, Outlier(**spec))
    assert bores(first) == bores(second)


def test_reset_replays_a_faulted_run_identically(config: MachineConfig) -> None:
    schedule = FaultSchedule(
        [Outlier(tag="BoreDiameter", probability=0.3, magnitude_sigma=4.0)], seed=42
    )
    sim = MachineSimulator(config, seed=42, faults=schedule)
    first = bores(sim.run_seconds(900.0))
    sim.reset()
    second = bores(sim.run_seconds(900.0))
    assert first == second


# --------------------------------------------------------------------------
# Tool wear
# --------------------------------------------------------------------------


def test_tool_wear_drifts_in_the_configured_direction(config: MachineConfig) -> None:
    samples = run(
        config, HOUR, ToolWear(tag="BoreDiameter", rate_per_hour=-0.045)
    )
    measured = bores(samples)
    early = float(np.mean(measured[:10]))
    late = float(np.mean(measured[-10:]))
    assert late < early - 0.03


def test_tool_wear_drift_size_matches_the_rate(config: MachineConfig) -> None:
    rate = -0.045
    samples = run(config, HOUR, ToolWear(tag="BoreDiameter", rate_per_hour=rate))
    measured = bores(samples)
    observed = float(np.mean(measured[-20:])) - float(np.mean(measured[:20]))
    assert observed == pytest.approx(rate, abs=0.008)


def test_tool_wear_does_nothing_before_it_starts(config: MachineConfig) -> None:
    samples = run(
        config, HOUR, ToolWear(tag="BoreDiameter", start_s=1800.0, rate_per_hour=-0.05)
    )
    before = [
        s.truth["BoreDiameter"]
        for s in samples
        if s.part_completed and s.t_s < 1800.0
    ]
    assert float(np.mean(before)) == pytest.approx(20.0, abs=0.004)


def test_tool_wear_produces_scrap_a_healthy_machine_does_not(
    config: MachineConfig,
) -> None:
    healthy = MachineSimulator(config, seed=42)
    healthy.run_seconds(HOUR)

    worn_schedule = FaultSchedule(
        [ToolWear(tag="BoreDiameter", start_s=600.0, rate_per_hour=-0.045)], seed=42
    )
    worn = MachineSimulator(config, seed=42, faults=worn_schedule)
    worn.run_seconds(HOUR)

    assert healthy.scrap_count == 0
    assert worn.scrap_count > 5


# --------------------------------------------------------------------------
# Mean shift
# --------------------------------------------------------------------------


def test_mean_shift_moves_the_average_by_the_right_amount(
    config: MachineConfig,
) -> None:
    spec = config.tag("BoreDiameter")
    samples = run(
        config, 2 * HOUR, MeanShift(tag="BoreDiameter", start_s=HOUR, shift_sigma=2.0)
    )
    before = [
        s.truth["BoreDiameter"] for s in samples if s.part_completed and s.t_s < HOUR
    ]
    after = [
        s.truth["BoreDiameter"] for s in samples if s.part_completed and s.t_s >= HOUR
    ]
    step = float(np.mean(after)) - float(np.mean(before))
    assert step == pytest.approx(2.0 * spec.std_dev, rel=0.25)


def test_mean_shift_does_not_change_the_spread(config: MachineConfig) -> None:
    spec = config.tag("BoreDiameter")
    samples = run(config, 2 * HOUR, MeanShift(tag="BoreDiameter", shift_sigma=3.0))
    assert float(np.std(bores(samples), ddof=1)) == pytest.approx(
        spec.std_dev, rel=0.2
    )


# --------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------


def test_outliers_appear_at_roughly_the_configured_rate(
    config: MachineConfig,
) -> None:
    spec = config.tag("BoreDiameter")
    samples = run(
        config,
        2 * HOUR,
        Outlier(tag="BoreDiameter", probability=0.1, magnitude_sigma=6.0),
    )
    measured = bores(samples)
    extreme = [b for b in measured if abs(b - spec.nominal) > 4 * spec.std_dev]
    assert 0.05 < len(extreme) / len(measured) < 0.16


def test_outliers_go_both_ways_when_asked(config: MachineConfig) -> None:
    spec = config.tag("BoreDiameter")
    samples = run(
        config,
        2 * HOUR,
        Outlier(tag="BoreDiameter", probability=0.2, magnitude_sigma=6.0),
    )
    measured = bores(samples)
    high = [b for b in measured if b - spec.nominal > 4 * spec.std_dev]
    low = [b for b in measured if spec.nominal - b > 4 * spec.std_dev]
    assert high and low


def test_zero_probability_produces_no_outliers(config: MachineConfig) -> None:
    plain = MachineSimulator(config, seed=42).run_seconds(HOUR)
    quiet = run(
        config, HOUR, Outlier(tag="BoreDiameter", probability=0.0, magnitude_sigma=9.0)
    )
    assert bores(plain) == bores(quiet)


# --------------------------------------------------------------------------
# Variance inflation
# --------------------------------------------------------------------------


def test_variance_inflation_widens_the_spread_without_moving_the_average(
    config: MachineConfig,
) -> None:
    spec = config.tag("BoreDiameter")
    samples = run(config, 2 * HOUR, VarianceInflation(tag="BoreDiameter", factor=2.5))
    measured = bores(samples)
    assert float(np.std(measured, ddof=1)) == pytest.approx(
        2.5 * spec.std_dev, rel=0.2
    )
    assert float(np.mean(measured)) == pytest.approx(spec.nominal, abs=0.003)


def test_a_factor_of_one_changes_nothing(config: MachineConfig) -> None:
    plain = MachineSimulator(config, seed=42).run_seconds(HOUR)
    same = run(config, HOUR, VarianceInflation(tag="BoreDiameter", factor=1.0))
    assert bores(plain) == bores(same)


# --------------------------------------------------------------------------
# Sensor faults: the reading lies, the part is fine
# --------------------------------------------------------------------------


def test_sensor_drift_moves_the_reading_but_not_the_part(
    config: MachineConfig,
) -> None:
    samples = run(
        config, 2 * HOUR, SensorDrift(tag="BoreDiameter", rate_per_hour=0.03)
    )
    reported = bores(samples, truth=False)
    real = bores(samples, truth=True)
    assert float(np.mean(reported[-20:])) > float(np.mean(reported[:20])) + 0.03
    assert float(np.mean(real)) == pytest.approx(20.0, abs=0.003)


def test_a_sensor_fault_produces_no_scrap(config: MachineConfig) -> None:
    """The whole point: alarms with an empty scrap bin means look at the gauge."""
    schedule = FaultSchedule(
        [SensorDrift(tag="BoreDiameter", rate_per_hour=0.08)], seed=42
    )
    sim = MachineSimulator(config, seed=42, faults=schedule)
    sim.run_seconds(2 * HOUR)
    assert sim.parts_completed > 400
    assert sim.scrap_count == 0


def test_a_stuck_sensor_reports_one_value_forever(config: MachineConfig) -> None:
    samples = run(config, HOUR, SensorStuck(tag="Torque", start_s=600.0))
    frozen = [s.values["Torque"] for s in samples if s.t_s >= 600.0]
    assert len(set(frozen)) == 1


def test_a_stuck_sensor_leaves_the_truth_moving(config: MachineConfig) -> None:
    samples = run(config, HOUR, SensorStuck(tag="Torque", start_s=600.0))
    after = [s for s in samples if s.t_s >= 600.0]
    assert len(set(actual(after, "Torque"))) > 1000


def test_a_stuck_sensor_recovers_when_the_fault_ends(config: MachineConfig) -> None:
    samples = run(
        config, HOUR, SensorStuck(tag="Torque", start_s=600.0, end_s=1200.0)
    )
    after_repair = [s.values["Torque"] for s in samples if s.t_s >= 1200.0]
    assert len(set(after_repair)) > 1000


def test_sensor_noise_widens_the_reading_not_the_part(config: MachineConfig) -> None:
    spec = config.tag("Torque")
    samples = run(config, HOUR, SensorNoise(tag="Torque", extra_sigma=2.0))
    reported_spread = float(np.std(published(samples, "Torque"), ddof=1))
    true_spread = float(np.std(actual(samples, "Torque"), ddof=1))
    assert true_spread == pytest.approx(spec.std_dev, rel=0.06)
    assert reported_spread > 2.0 * true_spread


# --------------------------------------------------------------------------
# Combined faults
# --------------------------------------------------------------------------


def test_two_faults_on_one_tag_stack(config: MachineConfig) -> None:
    spec = config.tag("BoreDiameter")
    samples = run(
        config,
        2 * HOUR,
        MeanShift(tag="BoreDiameter", shift_sigma=1.5),
        VarianceInflation(tag="BoreDiameter", factor=2.0),
    )
    measured = bores(samples)
    assert float(np.mean(measured)) == pytest.approx(
        spec.nominal + 1.5 * spec.std_dev, abs=0.004
    )
    assert float(np.std(measured, ddof=1)) == pytest.approx(
        2.0 * spec.std_dev, rel=0.2
    )


def test_a_process_fault_and_a_sensor_fault_can_run_together(
    config: MachineConfig,
) -> None:
    samples = run(
        config,
        HOUR,
        ToolWear(tag="BoreDiameter", rate_per_hour=-0.04),
        SensorNoise(tag="BoreDiameter", extra_sigma=1.5),
    )
    assert float(np.std(bores(samples, truth=False), ddof=1)) > float(
        np.std(bores(samples, truth=True), ddof=1)
    )


def test_active_faults_are_reported_on_every_sample(config: MachineConfig) -> None:
    samples = run(
        config,
        600.0,
        MeanShift(tag="Torque", start_s=200.0, end_s=400.0, shift_sigma=3.0),
    )
    assert samples[0].active_faults == ()
    assert not samples[0].is_faulted
    during = next(s for s in samples if 200.0 <= s.t_s < 400.0)
    assert during.active_faults == ("MeanShift(Torque)",)
    assert during.is_faulted
    after = next(s for s in samples if s.t_s >= 400.0)
    assert after.active_faults == ()


# --------------------------------------------------------------------------
# Time windows
# --------------------------------------------------------------------------


def test_is_active_covers_the_half_open_window() -> None:
    fault = MeanShift(tag="Torque", start_s=100.0, end_s=200.0, shift_sigma=1.0)
    assert not fault.is_active(99.9)
    assert fault.is_active(100.0)
    assert fault.is_active(199.9)
    assert not fault.is_active(200.0)


def test_a_fault_with_no_end_runs_forever() -> None:
    fault = MeanShift(tag="Torque", start_s=100.0, shift_sigma=1.0)
    assert fault.is_active(1_000_000.0)


def test_elapsed_hours_counts_from_the_start() -> None:
    fault = ToolWear(tag="Torque", start_s=1800.0, rate_per_hour=1.0)
    assert fault.elapsed_hours(1800.0) == 0.0
    assert fault.elapsed_hours(5400.0) == pytest.approx(1.0)
    assert fault.elapsed_hours(0.0) == 0.0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_a_fault_without_a_tag_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a tag name"):
        ToolWear(rate_per_hour=-0.01)


def test_a_negative_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="start_s"):
        ToolWear(tag="Torque", start_s=-5.0, rate_per_hour=-0.01)


def test_an_end_before_the_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="end_s"):
        MeanShift(tag="Torque", start_s=100.0, end_s=50.0, shift_sigma=1.0)


def test_an_impossible_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="probability"):
        Outlier(tag="Torque", probability=1.5, magnitude_sigma=3.0)


def test_a_non_positive_variance_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="factor"):
        VarianceInflation(tag="Torque", factor=0.0)


# --------------------------------------------------------------------------
# The effect arithmetic
# --------------------------------------------------------------------------


def test_process_effects_add_offsets_and_multiply_spreads() -> None:
    a = ProcessEffect(mean_offset=0.1, std_multiplier=2.0, spike=0.5)
    b = ProcessEffect(mean_offset=0.2, std_multiplier=3.0, spike=0.25)
    combined = a.combine(b)
    assert combined.mean_offset == pytest.approx(0.3)
    assert combined.std_multiplier == pytest.approx(6.0)
    assert combined.spike == pytest.approx(0.75)


def test_a_fresh_process_effect_is_neutral() -> None:
    assert ProcessEffect().is_neutral
    assert not ProcessEffect(mean_offset=0.1).is_neutral


def test_sensor_noise_combines_in_quadrature() -> None:
    a = SensorEffect(offset=1.0, extra_noise_std=3.0)
    b = SensorEffect(offset=2.0, extra_noise_std=4.0)
    combined = a.combine(b)
    assert combined.offset == pytest.approx(3.0)
    assert combined.extra_noise_std == pytest.approx(5.0)  # 3-4-5 triangle
    assert not combined.frozen


def test_frozen_wins_when_effects_combine() -> None:
    assert SensorEffect().combine(SensorEffect(frozen=True)).frozen


def test_a_fresh_sensor_effect_is_neutral() -> None:
    assert SensorEffect().is_neutral
    assert not SensorEffect(frozen=True).is_neutral


# --------------------------------------------------------------------------
# The schedule itself
# --------------------------------------------------------------------------


def test_an_empty_schedule_is_falsy() -> None:
    assert not FaultSchedule()
    assert len(FaultSchedule()) == 0


def test_a_schedule_reports_what_is_running_now() -> None:
    schedule = FaultSchedule(
        [
            ToolWear(tag="BoreDiameter", start_s=100.0, rate_per_hour=-0.01),
            SensorStuck(tag="Torque", start_s=500.0, end_s=800.0),
        ],
        seed=1,
    )
    assert schedule.labels_at(0.0) == []
    assert schedule.labels_at(200.0) == ["ToolWear(BoreDiameter)"]
    assert len(schedule.active_at(600.0)) == 2
    assert schedule.labels_at(900.0) == ["ToolWear(BoreDiameter)"]
    