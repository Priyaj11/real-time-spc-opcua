"""Tests for the machining station simulator."""

from __future__ import annotations

import numpy as np
import pytest

from spc_opcua.config import MachineConfig, load_config
from spc_opcua.simulator.machine import MachineSimulator, Sample, SimulatorSettings

TAG_NAMES = (
    "BoreDiameter",
    "Torque",
    "CycleTime",
    "Temperature",
    "Vibration",
    "ScrapCount",
)


@pytest.fixture
def config() -> MachineConfig:
    return load_config()


def values_of(samples: list[Sample], tag: str) -> list[float]:
    """Pull one tag out of a list of samples."""
    return [s[tag] for s in samples]


# --------------------------------------------------------------------------
# Reproducibility. Everything in Milestone 12 depends on this holding.
# --------------------------------------------------------------------------


def test_same_seed_produces_identical_runs(config: MachineConfig) -> None:
    a = MachineSimulator(config, seed=42).run(500)
    b = MachineSimulator(config, seed=42).run(500)
    for tag in TAG_NAMES:
        assert values_of(a, tag) == values_of(b, tag)


def test_different_seeds_produce_different_runs(config: MachineConfig) -> None:
    a = MachineSimulator(config, seed=42).run(500)
    b = MachineSimulator(config, seed=43).run(500)
    assert values_of(a, "Torque") != values_of(b, "Torque")


def test_reset_reproduces_the_same_run(config: MachineConfig) -> None:
    sim = MachineSimulator(config, seed=7)
    first = sim.run(300)
    sim.reset()
    second = sim.run(300)
    assert values_of(first, "BoreDiameter") == values_of(second, "BoreDiameter")


def test_reset_returns_the_clock_and_counters_to_zero(config: MachineConfig) -> None:
    sim = MachineSimulator(config, seed=7)
    sim.run_seconds(600.0)
    assert sim.parts_completed > 0
    sim.reset()
    assert sim.elapsed_s == 0.0
    assert sim.parts_completed == 0
    assert sim.scrap_count == 0


# --------------------------------------------------------------------------
# Shape and timing of the output
# --------------------------------------------------------------------------


def test_every_sample_carries_every_tag(config: MachineConfig) -> None:
    for sample in MachineSimulator(config, seed=1).run(50):
        assert tuple(sample.values.keys()) == TAG_NAMES
        assert all(isinstance(v, float) for v in sample.values.values())


def test_samples_are_spaced_one_sample_period_apart(config: MachineConfig) -> None:
    samples = MachineSimulator(config, seed=1).run(11)
    assert samples[0].t_s == 0.0
    for earlier, later in zip(samples, samples[1:]):
        assert later.t_s - earlier.t_s == pytest.approx(config.sample_period_s)


def test_sample_indices_count_up_from_zero(config: MachineConfig) -> None:
    samples = MachineSimulator(config, seed=1).run(20)
    assert [s.index for s in samples] == list(range(20))


def test_run_of_zero_samples_returns_nothing(config: MachineConfig) -> None:
    assert MachineSimulator(config, seed=1).run(0) == []


def test_negative_sample_count_is_rejected(config: MachineConfig) -> None:
    with pytest.raises(ValueError):
        MachineSimulator(config, seed=1).run(-1)


def test_run_seconds_produces_the_right_number_of_samples(
    config: MachineConfig,
) -> None:
    samples = MachineSimulator(config, seed=1).run_seconds(30.0)
    assert len(samples) == 300  # 30 seconds at 10 Hz


# --------------------------------------------------------------------------
# Per-part behaviour: a bore is measured once per part and held in between
# --------------------------------------------------------------------------


def test_bore_diameter_is_held_between_parts(config: MachineConfig) -> None:
    samples = MachineSimulator(config, seed=3).run_seconds(120.0)
    for earlier, later in zip(samples, samples[1:]):
        if not later.part_completed:
            assert later["BoreDiameter"] == earlier["BoreDiameter"]


def test_bore_diameter_changes_when_a_part_completes(config: MachineConfig) -> None:
    samples = MachineSimulator(config, seed=3).run_seconds(600.0)
    changes = [
        later["BoreDiameter"] != earlier["BoreDiameter"]
        for earlier, later in zip(samples, samples[1:])
        if later.part_completed
    ]
    assert changes and all(changes)


def test_parts_complete_at_roughly_the_configured_cycle_time(
    config: MachineConfig,
) -> None:
    sim = MachineSimulator(config, seed=3)
    sim.run_seconds(3600.0)
    expected = 3600.0 / config.tag("CycleTime").nominal  # about 300 parts
    assert sim.parts_completed == pytest.approx(expected, rel=0.05)


def test_continuous_tags_change_on_every_sample(config: MachineConfig) -> None:
    samples = MachineSimulator(config, seed=3).run(50)
    for tag in ("Torque", "Temperature", "Vibration"):
        series = values_of(samples, tag)
        assert len(set(series)) == len(series)


# --------------------------------------------------------------------------
# Statistical behaviour of a healthy process
# --------------------------------------------------------------------------


def test_bore_diameter_is_centred_on_nominal(config: MachineConfig) -> None:
    spec = config.tag("BoreDiameter")
    samples = MachineSimulator(config, seed=11).run_seconds(7200.0)
    bores = [s["BoreDiameter"] for s in samples if s.part_completed]
    assert len(bores) > 500
    assert float(np.mean(bores)) == pytest.approx(spec.nominal, abs=0.002)


def test_bore_diameter_spread_matches_the_configuration(
    config: MachineConfig,
) -> None:
    spec = config.tag("BoreDiameter")
    samples = MachineSimulator(config, seed=11).run_seconds(7200.0)
    bores = [s["BoreDiameter"] for s in samples if s.part_completed]
    assert float(np.std(bores, ddof=1)) == pytest.approx(spec.std_dev, rel=0.15)


def test_torque_matches_the_configuration(config: MachineConfig) -> None:
    spec = config.tag("Torque")
    torque = values_of(MachineSimulator(config, seed=11).run_seconds(3600.0), "Torque")
    assert float(np.mean(torque)) == pytest.approx(spec.nominal, abs=0.1)
    assert float(np.std(torque, ddof=1)) == pytest.approx(spec.std_dev, rel=0.05)


def test_vibration_total_spread_survives_the_torque_coupling(
    config: MachineConfig,
) -> None:
    """Coupling adds spread, so the independent noise is reduced to compensate."""
    spec = config.tag("Vibration")
    vib = values_of(
        MachineSimulator(config, seed=11).run_seconds(3600.0), "Vibration"
    )
    assert float(np.mean(vib)) == pytest.approx(spec.nominal, abs=0.03)
    assert float(np.std(vib, ddof=1)) == pytest.approx(spec.std_dev, rel=0.05)


def test_vibration_is_correlated_with_torque(config: MachineConfig) -> None:
    samples = MachineSimulator(config, seed=11).run_seconds(3600.0)
    correlation = float(
        np.corrcoef(values_of(samples, "Torque"), values_of(samples, "Vibration"))[0, 1]
    )
    assert correlation > 0.25


def test_cycle_time_is_always_positive(config: MachineConfig) -> None:
    samples = MachineSimulator(config, seed=11).run_seconds(1800.0)
    assert all(s["CycleTime"] > 0.0 for s in samples)


# --------------------------------------------------------------------------
# Temperature warm-up
# --------------------------------------------------------------------------


def test_cold_start_begins_near_ambient(config: MachineConfig) -> None:
    settings = SimulatorSettings(cold_start=True, ambient_temp_c=22.0)
    first = MachineSimulator(config, seed=5, settings=settings).step()
    assert first["Temperature"] == pytest.approx(22.0, abs=5.0)


def test_cold_start_warms_up_toward_nominal(config: MachineConfig) -> None:
    settings = SimulatorSettings(cold_start=True, ambient_temp_c=22.0)
    samples = MachineSimulator(config, seed=5, settings=settings).run_seconds(1800.0)
    early = float(np.mean(values_of(samples[:100], "Temperature")))
    late = float(np.mean(values_of(samples[-100:], "Temperature")))
    assert late > early + 10.0
    assert late == pytest.approx(config.tag("Temperature").nominal, abs=1.0)


def test_warm_start_begins_at_nominal(config: MachineConfig) -> None:
    settings = SimulatorSettings(cold_start=False)
    samples = MachineSimulator(config, seed=5, settings=settings).run_seconds(60.0)
    mean_temp = float(np.mean(values_of(samples, "Temperature")))
    assert mean_temp == pytest.approx(config.tag("Temperature").nominal, abs=0.5)


# --------------------------------------------------------------------------
# Scrap counting
# --------------------------------------------------------------------------


def test_scrap_count_never_decreases(config: MachineConfig) -> None:
    samples = MachineSimulator(config, seed=11).run_seconds(3600.0)
    counts = values_of(samples, "ScrapCount")
    assert all(b >= a for a, b in zip(counts, counts[1:]))


def test_a_healthy_process_scraps_almost_nothing(config: MachineConfig) -> None:
    """Bore tolerance sits about 4 sigma out, so scrap should be vanishingly rare."""
    sim = MachineSimulator(config, seed=11)
    sim.run_seconds(7200.0)
    assert sim.parts_completed > 500
    assert sim.scrap_count <= 1


def test_scrap_check_can_be_switched_off(config: MachineConfig) -> None:
    settings = SimulatorSettings(scrap_check_enabled=False)
    sim = MachineSimulator(config, seed=11, settings=settings)
    sim.run_seconds(3600.0)
    assert sim.scrap_count == 0
    