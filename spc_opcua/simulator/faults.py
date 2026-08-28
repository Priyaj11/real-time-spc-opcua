"""Fault injection for the machining station simulator.

A healthy machine is boring. Everything interesting in this project happens
when the machine goes wrong in a specific, known way, so that we can measure
whether the detection rules catch it.

Faults come in two families, and keeping them apart is the whole point of this
module:

  Process faults change the PART.
      Tool wear, mean shift, outliers, variance inflation. The metal really is
      the wrong size. Bad parts get made, scrap goes up, and the control chart
      is telling the truth.

  Sensor faults change only the READING.
      Drift, a stuck sensor, extra measurement noise. The part is fine. The
      number lying about it is not. Alarms fire and no scrap appears.

Telling those two apart from the chart alone is a genuine quality engineering
problem, and it is why every Sample carries both the published value and the
physical truth behind it.

Every fault is deterministic under a seed. The fault schedule owns a random
number generator separate from the machine's own, which means attaching an
inactive fault does not shift a single value in the healthy data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spc_opcua.config import TagSpec
from spc_opcua.simulator.distributions import make_rng

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600.0


# ---------------------------------------------------------------------------
# What a fault does, expressed as a small bundle of numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessEffect:
    """How a fault changes the part being made.

    Attributes:
        mean_offset: Added to the tag's nominal value, in engineering units.
        std_multiplier: Multiplies the tag's normal spread. 1.0 means unchanged.
        spike: A one-off jolt added to this single measurement only.
    """

    mean_offset: float = 0.0
    std_multiplier: float = 1.0
    spike: float = 0.0

    def combine(self, other: ProcessEffect) -> ProcessEffect:
        """Stack two effects. Offsets and spikes add, multipliers multiply."""
        return ProcessEffect(
            mean_offset=self.mean_offset + other.mean_offset,
            std_multiplier=self.std_multiplier * other.std_multiplier,
            spike=self.spike + other.spike,
        )

    @property
    def is_neutral(self) -> bool:
        """True when this effect changes nothing."""
        return (
            self.mean_offset == 0.0
            and self.std_multiplier == 1.0
            and self.spike == 0.0
        )


@dataclass(frozen=True)
class SensorEffect:
    """How a fault changes the reading without touching the part.

    Attributes:
        offset: Added to the reported value only.
        extra_noise_std: Extra random measurement noise on the reading only.
        frozen: When true the sensor reports its previous value forever.
    """

    offset: float = 0.0
    extra_noise_std: float = 0.0
    frozen: bool = False

    def combine(self, other: SensorEffect) -> SensorEffect:
        """Stack two effects. Offsets add, noise adds in quadrature, frozen wins."""
        combined_noise = float(
            np.hypot(self.extra_noise_std, other.extra_noise_std)
        )
        return SensorEffect(
            offset=self.offset + other.offset,
            extra_noise_std=combined_noise,
            frozen=self.frozen or other.frozen,
        )

    @property
    def is_neutral(self) -> bool:
        """True when this effect changes nothing."""
        return self.offset == 0.0 and self.extra_noise_std == 0.0 and not self.frozen


# ---------------------------------------------------------------------------
# The fault base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fault:
    """One named thing that can go wrong with one tag over a window of time.

    Attributes:
        tag: Which tag this fault affects.
        start_s: Simulated second the fault begins.
        end_s: Simulated second it stops, or None to run to the end.
    """

    tag: str = ""
    start_s: float = 0.0
    end_s: float | None = None

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError(f"{type(self).__name__} needs a tag name")
        if self.start_s < 0.0:
            raise ValueError(f"{self.label}: start_s must not be negative")
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError(f"{self.label}: end_s must be after start_s")

    @property
    def label(self) -> str:
        """Short human readable name, for logs and alarm listings."""
        return f"{type(self).__name__}({self.tag})"

    def is_active(self, t_s: float) -> bool:
        """True when the fault is running at this simulated time."""
        if t_s < self.start_s:
            return False
        return self.end_s is None or t_s < self.end_s

    def elapsed_s(self, t_s: float) -> float:
        """Seconds since the fault started, clamped at zero before it begins."""
        return max(0.0, t_s - self.start_s)

    def elapsed_hours(self, t_s: float) -> float:
        """Hours since the fault started."""
        return self.elapsed_s(t_s) / SECONDS_PER_HOUR

    def process_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> ProcessEffect:
        """How this fault changes the part. Neutral unless a subclass overrides."""
        return ProcessEffect()

    def sensor_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> SensorEffect:
        """How this fault changes the reading. Neutral unless a subclass overrides."""
        return SensorEffect()


# ---------------------------------------------------------------------------
# Process faults: the part really is wrong
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolWear(Fault):
    """A cutting tool blunting steadily, producing a slow one-way drift.

    The classic reason a control chart shows a trend rather than a jump. A worn
    boring insert cuts a smaller hole, so a negative rate is the physically
    normal case for BoreDiameter. A dull tool also has to push harder, so a
    realistic scenario pairs this with a positive rate on Torque.

    Attributes:
        rate_per_hour: Drift in engineering units per hour, signed.
    """

    rate_per_hour: float = 0.0

    def process_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> ProcessEffect:
        """Change the part itself. Nothing while the fault is inactive."""
        if not self.is_active(t_s):
            return ProcessEffect()
        return ProcessEffect(mean_offset=self.rate_per_hour * self.elapsed_hours(t_s))


@dataclass(frozen=True)
class MeanShift(Fault):
    """A sudden step to a new average and it stays there.

    Someone reset a tool offset, changed a fixture, or swapped material batch.
    The chart shows a clean jump rather than a slope.

    Attributes:
        shift_sigma: Size of the step, measured in the tag's own standard
            deviations, so the same number means the same severity on any tag.
    """

    shift_sigma: float = 0.0

    def process_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> ProcessEffect:
        """Change the part itself. Nothing while the fault is inactive."""
        if not self.is_active(t_s):
            return ProcessEffect()
        return ProcessEffect(mean_offset=self.shift_sigma * spec.std_dev)


@dataclass(frozen=True)
class Outlier(Fault):
    """Occasional wild single measurements against an otherwise normal process.

    A chip caught under the part, a momentary clamp slip. One measurement is
    badly wrong and the next is fine.

    Attributes:
        probability: Chance per measurement that a spike happens, 0 to 1.
        magnitude_sigma: Size of the spike in the tag's standard deviations.
        both_directions: Spike high or low at random, rather than always high.
    """

    probability: float = 0.0
    magnitude_sigma: float = 0.0
    both_directions: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"{self.label}: probability must be between 0 and 1")

    def process_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> ProcessEffect:
        """Change the part itself. Nothing while the fault is inactive."""
        if not self.is_active(t_s):
            return ProcessEffect()
        if rng.random() >= self.probability:
            return ProcessEffect()
        size = self.magnitude_sigma * spec.std_dev
        if self.both_directions and rng.random() < 0.5:
            size = -size
        return ProcessEffect(spike=size)


@dataclass(frozen=True)
class VarianceInflation(Fault):
    """The process stays centred but becomes much less consistent.

    A loosening fixture, a worn spindle bearing, coolant running out. The
    average looks fine, which is exactly why the R chart exists.

    Attributes:
        factor: Multiplies the tag's standard deviation. 2.0 doubles the spread.
    """

    factor: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.factor <= 0.0:
            raise ValueError(f"{self.label}: factor must be greater than zero")

    def process_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> ProcessEffect:
        """Change the part itself. Nothing while the fault is inactive."""
        if not self.is_active(t_s):
            return ProcessEffect()
        return ProcessEffect(std_multiplier=self.factor)


# ---------------------------------------------------------------------------
# Sensor faults: the part is fine, the reading is not
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorDrift(Fault):
    """A gauge slowly losing calibration.

    On a chart this is indistinguishable from tool wear, which is the point.
    The difference only shows up in the scrap count, because no bad parts are
    actually being made.

    Attributes:
        rate_per_hour: Reading error in engineering units per hour, signed.
    """

    rate_per_hour: float = 0.0

    def sensor_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> SensorEffect:
        """Change only the reading. Nothing while the fault is inactive."""
        if not self.is_active(t_s):
            return SensorEffect()
        return SensorEffect(offset=self.rate_per_hour * self.elapsed_hours(t_s))


@dataclass(frozen=True)
class SensorStuck(Fault):
    """A sensor that stops updating and repeats its last value.

    A dead gauge, a frozen fieldbus value, a cable pulled out. Produces a
    perfectly flat line, which no amount of process knowledge explains.
    """

    def sensor_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> SensorEffect:
        """Change only the reading. Nothing while the fault is inactive."""
        if not self.is_active(t_s):
            return SensorEffect()
        return SensorEffect(frozen=True)


@dataclass(frozen=True)
class SensorNoise(Fault):
    """Extra measurement noise on the reading, with the part unaffected.

    Electrical interference, a loose connector, a probe that needs cleaning.
    Widens the R chart while the parts themselves stay perfectly consistent.

    Attributes:
        extra_sigma: Added noise, measured in the tag's standard deviations.
    """

    extra_sigma: float = 0.0

    def sensor_effect(
        self, t_s: float, rng: np.random.Generator, spec: TagSpec
    ) -> SensorEffect:
        """Change only the reading. Nothing while the fault is inactive."""
        if not self.is_active(t_s):
            return SensorEffect()
        return SensorEffect(extra_noise_std=self.extra_sigma * spec.std_dev)


# ---------------------------------------------------------------------------
# The schedule that holds them all
# ---------------------------------------------------------------------------


_NEUTRAL_PROCESS = ProcessEffect()
_NEUTRAL_SENSOR = SensorEffect()


class FaultSchedule:
    """A collection of faults, with its own random number generator.

    The separate generator matters. If faults drew from the machine's own
    stream, attaching a fault that has not started yet would still shift every
    later value, and a healthy baseline could never be compared against a
    faulted run. Here, attaching an inactive fault changes nothing.

    Example:
        >>> schedule = FaultSchedule([ToolWear(tag="BoreDiameter",
        ...                                    start_s=600.0,
        ...                                    rate_per_hour=-0.02)], seed=7)
        >>> schedule.active_at(0.0)
        []
    """

    def __init__(self, faults: Sequence[Fault] = (), seed: int = 0) -> None:
        """Build a schedule.

        Args:
            faults: Every fault to apply. An empty schedule is a healthy machine.
            seed: Seed for the schedule's own generator.
        """
        self.faults: tuple[Fault, ...] = tuple(faults)
        self.seed = seed
        self._rng: np.random.Generator = make_rng(seed)

        # Group by tag once, so a draw never walks faults belonging to other
        # tags. On a healthy machine this dictionary is empty and every lookup
        # short-circuits.
        self._by_tag: dict[str, tuple[Fault, ...]] = {}
        for fault in self.faults:
            self._by_tag[fault.tag] = self._by_tag.get(fault.tag, ()) + (fault,)

        if self.faults:
            logger.info(
                "Fault schedule with %d fault(s): %s",
                len(self.faults),
                ", ".join(f.label for f in self.faults),
            )

    def __len__(self) -> int:
        return len(self.faults)

    def __bool__(self) -> bool:
        return bool(self.faults)

    def reset(self) -> None:
        """Rewind the generator so the same schedule replays identically."""
        self._rng = make_rng(self.seed)

    def active_at(self, t_s: float) -> list[Fault]:
        """Every fault running at this simulated time."""
        return [f for f in self.faults if f.is_active(t_s)]

    def labels_at(self, t_s: float) -> list[str]:
        """Names of every fault running at this simulated time."""
        return [f.label for f in self.active_at(t_s)]

    def process_effect(self, t_s: float, spec: TagSpec) -> ProcessEffect:
        """Combined effect of every active fault on the part for one tag."""
        relevant = self._by_tag.get(spec.name)
        if not relevant:
            return _NEUTRAL_PROCESS
        effect = ProcessEffect()
        for fault in relevant:
            effect = effect.combine(fault.process_effect(t_s, self._rng, spec))
        return effect

    def sensor_effect(self, t_s: float, spec: TagSpec) -> SensorEffect:
        """Combined effect of every active fault on the reading for one tag."""
        relevant = self._by_tag.get(spec.name)
        if not relevant:
            return _NEUTRAL_SENSOR
        effect = SensorEffect()
        for fault in relevant:
            effect = effect.combine(fault.sensor_effect(t_s, self._rng, spec))
        return effect

    def sensor_noise(self, std_dev: float) -> float:
        """Draw one extra measurement noise value from the schedule's generator."""
        if std_dev <= 0.0:
            return 0.0
        return float(self._rng.normal(loc=0.0, scale=std_dev))
