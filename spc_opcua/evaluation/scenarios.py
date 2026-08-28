"""The twelve fault scenarios the system is evaluated against.

Every scenario is a named, deterministic recipe: which fault, how severe, and
whether the part is genuinely wrong or only the reading is. They are defined
here as data rather than built inside the runner, so the catalogue can be read,
reviewed and argued with on its own.

Why the fault always starts after the baseline
----------------------------------------------
Control limits are computed from a stable period. If a fault were running
during that period, the limits would be calculated from faulted data, the fault
would be baked into what the chart considers normal, and detection would be
measured against a process that had already given up. So every scenario runs
healthy for the whole baseline, and the fault begins on the first monitored
part. START_S is that moment in simulated seconds.

Process faults versus sensor faults
-----------------------------------
Both must raise an alarm, because something really is wrong in both cases. Only
process faults can produce scrap. A drifting gauge makes the chart move while
every part it measures is fine, so counting its parts as scrap would flatter
the results. The `produces_scrap` flag keeps the two apart when the numbers are
totalled.
"""

from __future__ import annotations

from dataclasses import dataclass

from spc_opcua.simulator.faults import (
    Fault,
    MeanShift,
    Outlier,
    SensorDrift,
    SensorNoise,
    SensorStuck,
    ToolWear,
    VarianceInflation,
)

TAG = "BoreDiameter"

# How many subgroups of healthy production before the limits are frozen.
BASELINE_SUBGROUPS = 25

# How many subgroups are monitored after the fault begins. At five parts per
# subgroup and a twelve second cycle, 60 subgroups is 300 parts, one hour of
# production. A fault this system cannot see within an hour is a fault it has
# missed.
MONITOR_SUBGROUPS = 60

SUBGROUP_SIZE = 5
CYCLE_TIME_S = 12.0

# The simulated second the first monitored part is made, which is when every
# fault starts.
START_S = BASELINE_SUBGROUPS * SUBGROUP_SIZE * CYCLE_TIME_S


@dataclass(frozen=True)
class Scenario:
    """One named condition to evaluate the detector against.

    Attributes:
        name: Short identifier used in the results table.
        description: What is physically happening to the machine.
        faults: The faults injected, all starting at START_S.
        produces_scrap: True when the part itself is wrong. False for sensor
            faults, where the reading is wrong and the part is fine.
        expect_detection: True when an alarm is the correct outcome. False for
            the healthy scenario, where any alarm is a false alarm.
        severity: A rough label for reading the results table, not used in any
            calculation.
    """

    name: str
    description: str
    faults: tuple[Fault, ...] = ()
    produces_scrap: bool = True
    expect_detection: bool = True
    severity: str = "moderate"

    @property
    def is_healthy(self) -> bool:
        """True for the control scenario, which has nothing wrong with it."""
        return not self.faults

    @property
    def kind(self) -> str:
        """process, sensor, or healthy."""
        if self.is_healthy:
            return "healthy"
        return "process" if self.produces_scrap else "sensor"


def _process(name: str, description: str, fault: Fault, severity: str) -> Scenario:
    """A fault that changes the part itself, so it can produce scrap."""
    return Scenario(
        name=name, description=description, faults=(fault,), severity=severity
    )


def _sensor(name: str, description: str, fault: Fault, severity: str) -> Scenario:
    """A fault that changes only the reading, so it can never produce scrap."""
    return Scenario(
        name=name,
        description=description,
        faults=(fault,),
        produces_scrap=False,
        severity=severity,
    )


SCENARIOS: tuple[Scenario, ...] = (
    # ----------------------------------------------------------------- control
    Scenario(
        name="healthy",
        description="Nothing wrong. Every alarm here is a false alarm.",
        expect_detection=False,
        severity="none",
    ),
    # ------------------------------------------------- process: gradual drift
    _process(
        "tool-wear-slow",
        "Boring insert blunting at 0.02 mm/hour, under two sigma per hour.",
        ToolWear(tag=TAG, start_s=START_S, rate_per_hour=-0.02),
        "subtle",
    ),
    _process(
        "tool-wear-fast",
        "Boring insert blunting at 0.05 mm/hour, roughly four sigma per hour.",
        ToolWear(tag=TAG, start_s=START_S, rate_per_hour=-0.05),
        "moderate",
    ),
    # -------------------------------------------------- process: sudden shift
    _process(
        "mean-shift-1sigma",
        "A step of one sigma. Deliberately small: rule 1 alone cannot see this.",
        MeanShift(tag=TAG, start_s=START_S, shift_sigma=1.0),
        "subtle",
    ),
    _process(
        "mean-shift-2sigma",
        "A step of two sigma, for instance a fixture reset slightly off.",
        MeanShift(tag=TAG, start_s=START_S, shift_sigma=2.0),
        "moderate",
    ),
    _process(
        "mean-shift-3sigma",
        "A step of three sigma. A wrong offset entered on the machine.",
        MeanShift(tag=TAG, start_s=START_S, shift_sigma=3.0),
        "severe",
    ),
    # ---------------------------------------------- process: loss of control
    _process(
        "variance-2x",
        "Spread doubles with the average unmoved. A loose fixture or worn way.",
        VarianceInflation(tag=TAG, start_s=START_S, factor=2.0),
        "moderate",
    ),
    _process(
        "variance-3x",
        "Spread triples. The R chart should see this well before the X-bar.",
        VarianceInflation(tag=TAG, start_s=START_S, factor=3.0),
        "severe",
    ),
    _process(
        "outlier-burst",
        "Two per cent of parts land five sigma out. Chips under the workpiece.",
        Outlier(
            tag=TAG,
            start_s=START_S,
            probability=0.02,
            magnitude_sigma=5.0,
        ),
        "moderate",
    ),
    # ------------------------------------------------------------ sensor only
    _sensor(
        "sensor-drift",
        "Gauge losing calibration at 0.05 mm/hour. Parts are fine.",
        SensorDrift(tag=TAG, start_s=START_S, rate_per_hour=0.05),
        "moderate",
    ),
    _sensor(
        "sensor-stuck",
        "Gauge stops updating and repeats its last reading forever.",
        SensorStuck(tag=TAG, start_s=START_S),
        "severe",
    ),
    _sensor(
        "sensor-noise",
        "Two extra sigma of measurement noise. The parts are unchanged.",
        # extra_sigma is in units of the tag's own standard deviation, so 2.0
        # means twice the process noise added on top. Variances add, so the
        # reading's spread becomes sqrt(1 + 4) = 2.24 times the truth's: the
        # same widening as variance-2x, but with no bad parts behind it.
        SensorNoise(tag=TAG, start_s=START_S, extra_sigma=2.0),
        "moderate",
    ),
)

SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}


def scenario(name: str) -> Scenario:
    """Look one scenario up by name.

    Args:
        name: The scenario's short name.

    Returns:
        The matching Scenario.

    Raises:
        KeyError: with the list of valid names, if there is no such scenario.
    """
    try:
        return SCENARIOS_BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"Unknown scenario {name!r}. Choose from {sorted(SCENARIOS_BY_NAME)}."
        ) from None