"""X-bar and R control charts.

The two charts answer two different questions about the same subgroups.

    X-bar chart, plotting each subgroup's MEAN
        Has the process moved off centre?

    R chart, plotting each subgroup's RANGE
        Has the process become less consistent?

You need both. A machine can drift off target while producing beautifully
consistent parts, and it can sit perfectly on target while producing wildly
inconsistent ones. Either is a problem, and each chart is blind to the other's.

Order matters, and this trips people up. Read the R chart FIRST. The X-bar
limits are calculated from R-bar, so if the spread is out of control, the
limits themselves are built on sand and the X-bar chart cannot be trusted. Fix
consistency, then look at centring.

Two phases
----------
Phase 1, establishing the limits
    Collect a baseline from a process believed to be stable, twenty to
    twenty-five subgroups, and compute the limits from it. Nothing is being
    monitored yet. This is what fit() does.

Phase 2, monitoring
    Limits are now frozen. New subgroups are plotted against them, and anything
    unusual is a signal. This is what add() does.

Recomputing the limits every time a new point arrives is the single most common
mistake with control charts. Limits that chase the data can never detect a
drift, because they drift along with it.

Control limits are not specification limits
-------------------------------------------
Specification limits come from the customer and describe an acceptable part.
Control limits are calculated from the process and describe its own normal
behaviour. A point outside the control limits does not mean a bad part; it
means the machine is behaving unusually and someone should look. The two are
never drawn from each other, and putting specification limits on an X-bar chart
is a textbook error, because subgroup means are inherently less spread out than
individual parts.

Nothing in this module knows about OPC UA or the simulator. It takes subgroups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from spc_opcua.spc.constants import (
    RECOMMENDED_BASELINE_SUBGROUPS,
    ChartConstants,
    constants_for,
    sigma_from_mean_range,
)
from spc_opcua.spc.subgroups import Subgroup

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControlLimits:
    """Centre line and control limits for one chart, plus its sigma zones.

    The zones exist because the Nelson Rules in Milestone 9 are written in
    terms of them. Zone C is within one sigma of the centre, zone B between one
    and two, zone A between two and three.

    Attributes:
        center: The centre line.
        upper: Upper control limit, three sigma above the centre.
        lower: Lower control limit, three sigma below, floored at zero for
            ranges.
        sigma: Standard deviation of the plotted statistic, so that one third
            of the distance from the centre to a limit is one sigma.
        lower_is_floored: True when the lower limit was clipped to zero rather
            than being a genuine three sigma distance. Only ranges do this.
    """

    center: float
    upper: float
    lower: float
    sigma: float
    lower_is_floored: bool = False

    @property
    def width(self) -> float:
        """Distance from the lower limit to the upper limit."""
        return self.upper - self.lower

    def sigma_distance(self, value: float) -> float:
        """How many sigma above (positive) or below (negative) the centre."""
        if self.sigma <= 0.0:
            return 0.0
        return (value - self.center) / self.sigma

    def is_beyond_limits(self, value: float) -> bool:
        """True when a value falls outside the control limits."""
        return value > self.upper or value < self.lower

    def zone(self, value: float) -> str:
        """Name the sigma zone a value falls in.

        Returns:
            "C" within one sigma, "B" between one and two, "A" between two and
            three, "beyond" outside the limits. Prefixed with "+" above the
            centre and "-" below.
        """
        distance = self.sigma_distance(value)
        sign = "+" if distance >= 0 else "-"
        magnitude = abs(distance)
        if magnitude > 3.0:
            return "beyond"
        if magnitude > 2.0:
            return f"{sign}A"
        if magnitude > 1.0:
            return f"{sign}B"
        return f"{sign}C"


@dataclass(frozen=True)
class ChartPoint:
    """One subgroup, plotted against frozen limits.

    Attributes:
        index: Which subgroup this is.
        mean: Its average, plotted on the X-bar chart.
        range: Its largest minus smallest, plotted on the R chart.
        mean_sigma: How many sigma the mean sits from the X-bar centre line.
        range_sigma: How many sigma the range sits from the R centre line.
        mean_out_of_control: True when the mean is beyond the X-bar limits.
        range_out_of_control: True when the range is beyond the R limits.
        subgroup: The subgroup it came from, for traceability.
    """

    index: int
    mean: float
    range: float
    mean_sigma: float
    range_sigma: float
    mean_out_of_control: bool
    range_out_of_control: bool
    subgroup: Subgroup | None = None

    @property
    def out_of_control(self) -> bool:
        """True when either chart flags this subgroup."""
        return self.mean_out_of_control or self.range_out_of_control

    def as_row(self) -> dict[str, object]:
        """Flatten into one dictionary, suitable for a table or a CSV row."""
        return {
            "subgroup": self.index,
            "mean": self.mean,
            "range": self.range,
            "mean_sigma": self.mean_sigma,
            "range_sigma": self.range_sigma,
            "mean_ooc": self.mean_out_of_control,
            "range_ooc": self.range_out_of_control,
        }


@dataclass(frozen=True)
class ChartLimits:
    """Everything computed from one baseline.

    Attributes:
        subgroup_size: Parts per subgroup, called n.
        baseline_count: How many subgroups the limits were built from.
        grand_mean: Average of every subgroup mean. Written x-double-bar.
        mean_range: Average of every subgroup range. Written R-bar.
        sigma_within: Short-term process spread, estimated as R-bar over d2.
        xbar: Limits for the chart of subgroup means.
        r: Limits for the chart of subgroup ranges.
        constants: The chart constants used.
    """

    subgroup_size: int
    baseline_count: int
    grand_mean: float
    mean_range: float
    sigma_within: float
    xbar: ControlLimits
    r: ControlLimits
    constants: ChartConstants

    @property
    def is_well_founded(self) -> bool:
        """True when the baseline was big enough to trust the limits."""
        return self.baseline_count >= RECOMMENDED_BASELINE_SUBGROUPS

    def describe(self) -> str:
        """A short human readable report of the limits."""
        n = self.subgroup_size
        note = "" if self.is_well_founded else "  (below the recommended 20)"
        return (
            f"Baseline       : {self.baseline_count} subgroups of {n}{note}\n"
            f"Grand mean     : {self.grand_mean:.5f}\n"
            f"Mean range     : {self.mean_range:.5f}\n"
            f"Sigma (within) : {self.sigma_within:.5f}  "
            f"(R-bar / d2, d2 = {self.constants.d2})\n"
            f"A2             : {self.constants.a2:.4f}\n"
            f"X-bar limits   : {self.xbar.lower:.5f} .. "
            f"{self.xbar.center:.5f} .. {self.xbar.upper:.5f}\n"
            f"R limits       : {self.r.lower:.5f} .. "
            f"{self.r.center:.5f} .. {self.r.upper:.5f}"
        )


def compute_limits(subgroups: Sequence[Subgroup]) -> ChartLimits:
    """Calculate X-bar and R control limits from a baseline of subgroups.

    Args:
        subgroups: The baseline. Every subgroup must be the same size.

    Returns:
        The limits, ready to freeze and monitor against.

    Raises:
        ValueError: if the baseline is empty or the sizes are inconsistent.
    """
    if not subgroups:
        raise ValueError("Control limits need at least one baseline subgroup")

    sizes = {group.size for group in subgroups}
    if len(sizes) > 1:
        raise ValueError(
            f"Every baseline subgroup must be the same size, found {sorted(sizes)}"
        )

    n = sizes.pop()
    constants = constants_for(n)

    grand_mean = sum(group.mean for group in subgroups) / len(subgroups)
    mean_range = sum(group.range for group in subgroups) / len(subgroups)
    sigma_within = sigma_from_mean_range(mean_range, n)

    # A2 * R-bar is three standard deviations of a subgroup MEAN, which is
    # sigma / sqrt(n), not sigma.
    half_width = constants.a2 * mean_range
    xbar = ControlLimits(
        center=grand_mean,
        upper=grand_mean + half_width,
        lower=grand_mean - half_width,
        sigma=half_width / 3.0,
    )

    upper_range = constants.d4_upper * mean_range
    lower_range = constants.d3_lower * mean_range
    r = ControlLimits(
        center=mean_range,
        upper=upper_range,
        lower=lower_range,
        # Three sigma of the range is the distance to the UPPER limit, which is
        # always a genuine three sigma even when the lower one has been floored.
        sigma=(upper_range - mean_range) / 3.0,
        lower_is_floored=constants.d3_lower == 0.0,
    )

    limits = ChartLimits(
        subgroup_size=n,
        baseline_count=len(subgroups),
        grand_mean=grand_mean,
        mean_range=mean_range,
        sigma_within=sigma_within,
        xbar=xbar,
        r=r,
        constants=constants,
    )

    if not limits.is_well_founded:
        logger.warning(
            "Control limits built from only %d subgroups; %d or more is the "
            "usual minimum before the limits are worth trusting",
            len(subgroups),
            RECOMMENDED_BASELINE_SUBGROUPS,
        )
    return limits


class XbarRChart:
    """An X-bar and R chart pair with frozen limits.

    Example:
        >>> from spc_opcua.spc.subgroups import subgroups_from_values
        >>> baseline = subgroups_from_values([20.0, 20.01, 19.99] * 20, size=3)
        >>> chart = XbarRChart.fit(baseline)
        >>> chart.limits.subgroup_size
        3
    """

    def __init__(self, limits: ChartLimits) -> None:
        """Build a chart from already computed limits.

        Args:
            limits: Limits from a baseline. Frozen from here on.
        """
        self.limits = limits
        self._points: list[ChartPoint] = []

    @classmethod
    def fit(cls, baseline: Sequence[Subgroup]) -> "XbarRChart":
        """Phase 1: compute limits from a baseline and freeze them.

        Args:
            baseline: Subgroups from a process believed to be stable.

        Returns:
            A chart ready to monitor with.
        """
        return cls(compute_limits(baseline))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def points(self) -> tuple[ChartPoint, ...]:
        """Every subgroup plotted so far, in order."""
        return tuple(self._points)

    @property
    def out_of_control_points(self) -> tuple[ChartPoint, ...]:
        """Only the points either chart flagged."""
        return tuple(p for p in self._points if p.out_of_control)

    @property
    def first_signal(self) -> ChartPoint | None:
        """The earliest flagged point, or None if the process stayed in control."""
        flagged = self.out_of_control_points
        return flagged[0] if flagged else None

    @property
    def range_is_in_control(self) -> bool:
        """True when no plotted range has gone outside the R limits.

        Read this before trusting anything on the X-bar chart. The X-bar limits
        are derived from R-bar, so an out-of-control range means the X-bar
        limits themselves are not meaningful.
        """
        return not any(p.range_out_of_control for p in self._points)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def evaluate(self, subgroup: Subgroup, index: int | None = None) -> ChartPoint:
        """Plot one subgroup against the frozen limits, without recording it.

        Args:
            subgroup: The subgroup to plot.
            index: Position on the chart. Uses the subgroup's own index if
                omitted.

        Returns:
            The resulting chart point.

        Raises:
            ValueError: if the subgroup is a different size from the baseline.
        """
        if subgroup.size != self.limits.subgroup_size:
            raise ValueError(
                f"Subgroup of {subgroup.size} cannot be plotted against limits "
                f"built for subgroups of {self.limits.subgroup_size}"
            )
        mean = subgroup.mean
        spread = subgroup.range
        return ChartPoint(
            index=subgroup.index if index is None else index,
            mean=mean,
            range=spread,
            mean_sigma=self.limits.xbar.sigma_distance(mean),
            range_sigma=self.limits.r.sigma_distance(spread),
            mean_out_of_control=self.limits.xbar.is_beyond_limits(mean),
            range_out_of_control=self.limits.r.is_beyond_limits(spread),
            subgroup=subgroup,
        )

    def add(self, subgroup: Subgroup) -> ChartPoint:
        """Phase 2: plot one subgroup and record it on the chart."""
        point = self.evaluate(subgroup, index=len(self._points))
        self._points.append(point)
        if point.range_out_of_control:
            logger.info(
                "Subgroup %d range %.5f outside R limits %.5f to %.5f",
                point.index,
                point.range,
                self.limits.r.lower,
                self.limits.r.upper,
            )
        if point.mean_out_of_control:
            logger.info(
                "Subgroup %d mean %.5f outside X-bar limits %.5f to %.5f",
                point.index,
                point.mean,
                self.limits.xbar.lower,
                self.limits.xbar.upper,
            )
        return point

    def add_many(self, subgroups: Iterable[Subgroup]) -> list[ChartPoint]:
        """Plot many subgroups in order."""
        return [self.add(group) for group in subgroups]

    def reset(self) -> None:
        """Clear the plotted points, keeping the limits."""
        self._points.clear()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def means(self) -> list[float]:
        """Every plotted subgroup mean, in order."""
        return [p.mean for p in self._points]

    def ranges(self) -> list[float]:
        """Every plotted subgroup range, in order."""
        return [p.range for p in self._points]

    def to_frame(self):
        """Every plotted point as a pandas table."""
        import pandas as pd

        if not self._points:
            return pd.DataFrame(
                columns=["subgroup", "mean", "range", "mean_sigma", "range_sigma"]
            )
        return pd.DataFrame([p.as_row() for p in self._points])


# ---------------------------------------------------------------------------
# Command line demonstration
# ---------------------------------------------------------------------------


def _bore_values(faults, seed: int, parts: int) -> list[float]:
    """Run the simulator offline and return one bore measurement per part."""
    from spc_opcua.config import load_config
    from spc_opcua.simulator.machine import MachineSimulator

    simulator = MachineSimulator(load_config(), seed=seed, faults=faults)
    values: list[float] = []
    while len(values) < parts:
        sample = simulator.step()
        if sample.part_completed:
            values.append(sample.values["BoreDiameter"])
    return values


def main() -> None:
    """Fit limits on a healthy baseline, then watch a tool-wear run against them."""
    from spc_opcua.config import load_config
    from spc_opcua.logging_setup import configure_logging
    from spc_opcua.simulator.faults import FaultSchedule, ToolWear
    from spc_opcua.spc.subgroups import subgroups_from_values

    configure_logging(level="WARNING")
    config = load_config()
    spec = config.tag("BoreDiameter")
    n = config.subgroup_size

    # Phase 1: a healthy baseline, 25 subgroups of 5.
    baseline_values = _bore_values(FaultSchedule(), seed=1, parts=25 * n)
    baseline = subgroups_from_values(baseline_values, n, tag="BoreDiameter")
    chart = XbarRChart.fit(baseline)

    print("\nPHASE 1  establishing limits from a healthy baseline")
    print("-" * 62)
    print(chart.limits.describe())
    print(
        f"\nSpecification  : {spec.lsl:.3f} .. {spec.nominal:.3f} .. "
        f"{spec.usl:.3f} {spec.units}   (from the customer)"
    )
    print(
        f"X-bar limits   : {chart.limits.xbar.lower:.5f} .. "
        f"{chart.limits.xbar.upper:.5f}   (from the process)"
    )
    print(
        "Notice the control limits are far TIGHTER than the specification.\n"
        "Subgroup means are less spread out than single parts, by sqrt(n)."
    )

    # Phase 2: the same machine, now with a worn tool, plotted against those
    # frozen limits.
    wear = FaultSchedule(
        [ToolWear(tag="BoreDiameter", start_s=0.0, rate_per_hour=-0.05)], seed=1
    )
    watch_values = _bore_values(wear, seed=1, parts=40 * n)
    watch = subgroups_from_values(watch_values, n, tag="BoreDiameter")
    chart.add_many(watch)

    print("\n\nPHASE 2  monitoring a tool-wear run against the frozen limits")
    print("-" * 62)
    print(f"{'SUB':>4}{'MEAN':>11}{'RANGE':>10}{'X-SIGMA':>10}{'ZONE':>7}  FLAG")
    for point in chart.points:
        flag = ""
        if point.range_out_of_control:
            flag += "R-OUT "
        if point.mean_out_of_control:
            flag += "XBAR-OUT"
        print(
            f"{point.index:>4}{point.mean:>11.5f}{point.range:>10.5f}"
            f"{point.mean_sigma:>10.2f}{chart.limits.xbar.zone(point.mean):>7}"
            f"  {flag}"
        )

    # The headline comparison: chart signal versus the first bad part.
    first_signal = chart.first_signal
    first_bad_part = next(
        (
            i
            for i, value in enumerate(watch_values)
            if value < spec.lsl or value > spec.usl
        ),
        None,
    )

    print("\n" + "-" * 62)
    print(f"R chart in control throughout : {chart.range_is_in_control}")
    if first_signal is None:
        print("No X-bar or R signal in this run.")
    else:
        signal_part = (first_signal.index + 1) * n
        print(
            f"First control chart signal    : subgroup {first_signal.index}"
            f", after part {signal_part}"
        )
    if first_bad_part is None:
        print("No part went outside specification during this run.")
    elif first_signal is not None:
        signal_part = (first_signal.index + 1) * n
        print(f"First out-of-specification part: part {first_bad_part + 1}")
        print(
            f"\nDetected {first_bad_part + 1 - signal_part} parts "
            f"({(first_bad_part + 1 - signal_part) // n} subgroups) "
            "before the first defect."
        )
    print()


if __name__ == "__main__":
    main()