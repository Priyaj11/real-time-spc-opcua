"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from spc_opcua.config import ConfigError, MachineConfig, TagSpec, load_config

EXPECTED_TAGS = (
    "BoreDiameter",
    "Torque",
    "CycleTime",
    "Temperature",
    "Vibration",
    "ScrapCount",
)


def test_default_config_loads() -> None:
    config = load_config()
    assert isinstance(config, MachineConfig)
    assert config.name == "BORE-01"


def test_default_config_has_all_six_tags() -> None:
    config = load_config()
    assert config.tag_names == EXPECTED_TAGS


def test_sample_period_matches_sample_rate() -> None:
    config = load_config()
    assert config.sample_rate_hz == pytest.approx(10.0)
    assert config.sample_period_s == pytest.approx(0.1)


def test_bore_diameter_spec_limits() -> None:
    bore = load_config().tag("BoreDiameter")
    assert bore.lsl == pytest.approx(19.950)
    assert bore.usl == pytest.approx(20.050)
    assert bore.nominal == pytest.approx(20.000)
    assert bore.is_two_sided
    assert bore.tolerance_width == pytest.approx(0.100)


def test_scrap_count_is_a_counter_without_spec_limits() -> None:
    scrap = load_config().tag("ScrapCount")
    assert scrap.is_counter
    assert not scrap.has_spec_limits
    assert scrap.tolerance_width is None


def test_unknown_tag_raises_key_error() -> None:
    config = load_config()
    with pytest.raises(KeyError):
        config.tag("NoSuchTag")


def test_nominal_outside_spec_limits_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must sit between"):
        TagSpec(
            name="Bad", units="mm", nominal=99.0, std_dev=0.1, lsl=1.0, usl=2.0
        ).validate()


def test_inverted_spec_limits_are_rejected() -> None:
    with pytest.raises(ConfigError, match="must be below usl"):
        TagSpec(
            name="Bad", units="mm", nominal=1.5, std_dev=0.1, lsl=2.0, usl=1.0
        ).validate()


def test_negative_std_dev_is_rejected() -> None:
    with pytest.raises(ConfigError, match="std_dev"):
        TagSpec(name="Bad", units="mm", nominal=1.0, std_dev=-0.5).validate()


def test_subgroup_size_below_two_is_rejected() -> None:
    good_tag = TagSpec(name="A", units="mm", nominal=1.0, std_dev=0.1)
    with pytest.raises(ConfigError, match="subgroup_size"):
        MachineConfig(
            name="X",
            description="",
            sample_rate_hz=10.0,
            subgroup_size=1,
            random_seed=0,
            tags=(good_tag,),
        ).validate()


def test_duplicate_tag_names_are_rejected() -> None:
    tag = TagSpec(name="A", units="mm", nominal=1.0, std_dev=0.1)
    with pytest.raises(ConfigError, match="Duplicate tag name"):
        MachineConfig(
            name="X",
            description="",
            sample_rate_hz=10.0,
            subgroup_size=5,
            random_seed=0,
            tags=(tag, tag),
        ).validate()


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("machine: [this is not a mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_missing_tags_section_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "no_tags.yaml"
    bad.write_text(
        "machine:\n  name: X\n  sample_rate_hz: 10\n  subgroup_size: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="tags"):
        load_config(bad)