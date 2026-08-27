"""Configuration loading and validation for the SPC over OPC UA project.

Everything the rest of the system needs to know about the machine lives in a
YAML file (config/machine.yaml). This module turns that file into typed Python
objects and refuses to load a file that does not make engineering sense.

Loading configuration this way, instead of hard-coding numbers, is what lets us
later run the exact same code against a different machine definition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# The project root is two directories above this file:
# <root>/spc_opcua/config.py -> parents[1] is <root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "machine.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed, or impossible."""


@dataclass(frozen=True)
class TagSpec:
    """One measured or counted value that the machine publishes.

    Attributes:
        name: Tag name, used as the OPC UA variable name.
        units: Engineering units, for display only.
        nominal: The target value the process aims at.
        std_dev: Short-term standard deviation of the healthy process.
        lsl: Lower specification limit, or None when the tag has no lower limit.
        usl: Upper specification limit, or None when the tag has no upper limit.
        decimals: How many decimal places to show for this tag.
        is_counter: True for integer counters such as ScrapCount, which are
            counted rather than measured and are not put on a control chart.
        description: Human readable note about what the tag represents.
    """

    name: str
    units: str
    nominal: float
    std_dev: float
    lsl: float | None = None
    usl: float | None = None
    decimals: int = 3
    is_counter: bool = False
    description: str = ""

    @property
    def has_spec_limits(self) -> bool:
        """True when at least one specification limit is defined."""
        return self.lsl is not None or self.usl is not None

    @property
    def is_two_sided(self) -> bool:
        """True when both a lower and an upper specification limit exist."""
        return self.lsl is not None and self.usl is not None

    @property
    def tolerance_width(self) -> float | None:
        """Distance from the lower to the upper specification limit."""
        if not self.is_two_sided:
            return None
        assert self.usl is not None and self.lsl is not None
        return self.usl - self.lsl

    def validate(self) -> None:
        """Check that this tag definition is physically sensible.

        Raises:
            ConfigError: if any limit or spread value is impossible.
        """
        if not self.name:
            raise ConfigError("A tag has an empty name")
        if self.std_dev < 0:
            raise ConfigError(f"{self.name}: std_dev must not be negative")
        if self.decimals < 0:
            raise ConfigError(f"{self.name}: decimals must not be negative")
        if self.is_two_sided:
            assert self.lsl is not None and self.usl is not None
            if self.lsl >= self.usl:
                raise ConfigError(
                    f"{self.name}: lsl ({self.lsl}) must be below usl ({self.usl})"
                )
            if not (self.lsl < self.nominal < self.usl):
                raise ConfigError(
                    f"{self.name}: nominal ({self.nominal}) must sit between "
                    f"lsl ({self.lsl}) and usl ({self.usl})"
                )
        if self.lsl is not None and not self.is_two_sided and self.nominal <= self.lsl:
            raise ConfigError(
                f"{self.name}: nominal ({self.nominal}) must be above lsl ({self.lsl})"
            )
        if self.usl is not None and not self.is_two_sided and self.nominal >= self.usl:
            raise ConfigError(
                f"{self.name}: nominal ({self.nominal}) must be below usl ({self.usl})"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TagSpec:
        """Build a TagSpec from one YAML mapping, then validate it."""
        try:
            spec = cls(
                name=str(raw["name"]),
                units=str(raw.get("units", "")),
                nominal=float(raw["nominal"]),
                std_dev=float(raw["std_dev"]),
                lsl=None if raw.get("lsl") is None else float(raw["lsl"]),
                usl=None if raw.get("usl") is None else float(raw["usl"]),
                decimals=int(raw.get("decimals", 3)),
                is_counter=bool(raw.get("is_counter", False)),
                description=str(raw.get("description", "")),
            )
        except KeyError as exc:
            raise ConfigError(f"Tag entry is missing required key: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Tag entry has a bad value: {exc}") from exc
        spec.validate()
        return spec


@dataclass(frozen=True)
class MachineConfig:
    """The complete machine definition: identity, timing, and every tag.

    Attributes:
        name: Short machine identifier, for example BORE-01.
        description: Human readable description of the station.
        sample_rate_hz: How many samples per second the machine publishes.
        subgroup_size: How many consecutive samples form one SPC subgroup.
        random_seed: Seed for the random number generator, so runs repeat exactly.
        tags: Every tag the machine publishes, in file order.
    """

    name: str
    description: str
    sample_rate_hz: float
    subgroup_size: int
    random_seed: int
    tags: tuple[TagSpec, ...] = field(default_factory=tuple)

    @property
    def sample_period_s(self) -> float:
        """Seconds between two samples, the inverse of the sample rate."""
        return 1.0 / self.sample_rate_hz

    @property
    def tag_names(self) -> tuple[str, ...]:
        """Every tag name, in file order."""
        return tuple(tag.name for tag in self.tags)

    def tag(self, name: str) -> TagSpec:
        """Look up one tag by name.

        Raises:
            KeyError: if no tag with that name exists.
        """
        for candidate in self.tags:
            if candidate.name == name:
                return candidate
        raise KeyError(f"No tag named {name!r}. Known tags: {self.tag_names}")

    def validate(self) -> None:
        """Check the machine level settings and every tag.

        Raises:
            ConfigError: if any setting is impossible.
        """
        if self.sample_rate_hz <= 0:
            raise ConfigError("sample_rate_hz must be greater than zero")
        if self.subgroup_size < 2:
            raise ConfigError("subgroup_size must be at least 2")
        if self.subgroup_size > 25:
            raise ConfigError("subgroup_size above 25 has no standard SPC constants")
        if not self.tags:
            raise ConfigError("At least one tag must be defined")
        seen: set[str] = set()
        for tag in self.tags:
            if tag.name in seen:
                raise ConfigError(f"Duplicate tag name: {tag.name}")
            seen.add(tag.name)
            tag.validate()


def load_config(path: Path | str | None = None) -> MachineConfig:
    """Read the YAML configuration file and return a validated MachineConfig.

    Args:
        path: Path to the YAML file. Defaults to config/machine.yaml at the
            project root.

    Returns:
        A fully validated MachineConfig.

    Raises:
        ConfigError: if the file is missing, unreadable, or invalid.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    logger.debug("Loading configuration from %s", config_path)

    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")

    machine_section = raw.get("machine")
    if not isinstance(machine_section, dict):
        raise ConfigError(f"{config_path} is missing a 'machine:' section")

    tag_section = raw.get("tags")
    if not isinstance(tag_section, list) or not tag_section:
        raise ConfigError(f"{config_path} is missing a non-empty 'tags:' list")

    try:
        config = MachineConfig(
            name=str(machine_section["name"]),
            description=str(machine_section.get("description", "")),
            sample_rate_hz=float(machine_section["sample_rate_hz"]),
            subgroup_size=int(machine_section["subgroup_size"]),
            random_seed=int(machine_section.get("random_seed", 0)),
            tags=tuple(TagSpec.from_dict(entry) for entry in tag_section),
        )
    except KeyError as exc:
        raise ConfigError(f"'machine:' section is missing required key: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'machine:' section has a bad value: {exc}") from exc

    config.validate()
    logger.info(
        "Loaded machine %s with %d tags at %.1f Hz, subgroup size %d",
        config.name,
        len(config.tags),
        config.sample_rate_hz,
        config.subgroup_size,
    )
    return config


def format_config(config: MachineConfig) -> str:
    """Render the configuration as a plain text table for the console."""
    header = (
        f"Machine   : {config.name}\n"
        f"About     : {config.description}\n"
        f"Rate      : {config.sample_rate_hz:g} Hz "
        f"({config.sample_period_s * 1000:.0f} ms between samples)\n"
        f"Subgroup  : {config.subgroup_size} samples per SPC point\n"
        f"Seed      : {config.random_seed}\n"
    )
    columns = f"\n{'TAG':<14}{'UNITS':<8}{'NOMINAL':>12}{'SIGMA':>10}{'LSL':>12}{'USL':>12}\n"
    rule = "-" * 68 + "\n"
    rows = ""
    for tag in config.tags:
        lsl = "none" if tag.lsl is None else f"{tag.lsl:.{tag.decimals}f}"
        usl = "none" if tag.usl is None else f"{tag.usl:.{tag.decimals}f}"
        rows += (
            f"{tag.name:<14}{tag.units:<8}"
            f"{tag.nominal:>12.{tag.decimals}f}"
            f"{tag.std_dev:>10.{tag.decimals}f}"
            f"{lsl:>12}{usl:>12}\n"
        )
    return header + columns + rule + rows


def main() -> None:
    """Load the default configuration and print it. Entry point for a smoke test."""
    from spc_opcua.logging_setup import configure_logging

    configure_logging()
    config = load_config()
    print()
    print(format_config(config))


if __name__ == "__main__":
    main()
    