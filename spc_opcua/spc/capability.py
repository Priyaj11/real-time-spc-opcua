"""Process capability: Cp, Cpk, Pp and Ppk.

The control charts in Milestone 7 answer one question: is the process behaving
consistently? Capability answers a completely different one: is what it is
consistently doing good enough for the customer?

Those are independent, and the four combinations are all real:

    In control and capable
        The goal. Predictable, and comfortably inside tolerance.

    In control but not capable
        The machine is doing exactly what it always does, and what it always
        does is make scrap. No amount of chart-watching fixes this. It needs a
        better machine, a better fixture, or a wider tolerance.

    Out of control but capable
        Something changed, but the tolerance is wide enough to absorb it. Find
        it now, because the next change may not be so forgiving.

    Out of control and not capable
        Everything is on fire.

Capability is only meaningful on a stable process. If the machine is drifting,
its "mean" and "spread" are not describing anything that will still be true
tomorrow, and a capability index computed from them is a number about the past
dressed up as a prediction. Always read the control charts first.

The indices
-----------
    Cp  = (USL - LSL) / (6 * sigma)
        POTENTIAL capability. How well the spread fits the tolerance, ignoring
        where the process is centred. A process could have a wonderful Cp while
        sitting entirely outside the tolerance.

    Cpu = (USL - mean) / (3 * sigma)      Cpl = (mean - LSL) / (3 * sigma)
    Cpk = the smaller of the two
        ACTUAL capability. Distance from the mean to the nearest limit,
        measured in three-sigma units. Being off centre can only make it worse,
        so Cpk is never greater than Cp, and they are equal only when the
        process sits exactly in the middle.

    Pp and Ppk
        The same formulas with a different sigma. Cp and Cpk use the WITHIN-
        subgroup spread, R-bar over d2, which is short-term noise. Pp and Ppk
        use the overall standard deviation of every individual measurement,
        which also contains any drift between subgroups.

        The gap between them is diagnostic and very often the most useful
        number on the page. Cpk much larger than Ppk means the machine is
        capable moment to moment but wandering over the shift. That is a
        setup, tooling or thermal problem, not a machine capability problem.

Rules of thumb, and they are only that: 1.33 is the usual industry floor, 1.67
is good, 2.0 is what people mean by a "six sigma" process. Below 1.0 the
process is actively producing scrap.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from spc_opcua.config import TagSpec
from spc_opcua.spc.constants import sigma_from_mean_range
from spc_opcua.spc.subgroups import Subgroup

logger = logging.getLogger(__name__)

# The usual interpretation bands. Not a standard, but what every quality
# department means when they quote a number.
MINIMUM_ACCEPTABLE_CPK = 1.33
GOOD_CPK = 1.67
EXCELLENT_CPK = 2.00


def normal_tail(z: float) -> float:
    """Fraction of a normal distribution beyond z standard deviations.

    Uses the complementary error function, which stays accurate far into the
    tail where 1 minus the cumulative distribution would lose all its precision
    to floating point rounding.

    Args:
        z: Distance from the mean, in standard deviations.

    Returns:
        The one-sided tail area, between 0 and 1.
    """
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass(frozen=True)
class Capability:
    """One capability assessment of one tag.

    Attributes:
        tag: Which measurement this describes.
        sample_size: How many individual parts it was computed from.
        subgroup_count: How many subgroups those parts formed.
        mean: Process average.
        sigma_within: Short-term spread, R-bar over d2. Drives Cp and Cpk.
        sigma_overall: Spread of every individual value. Drives Pp and Ppk.
        lsl: Lower specification limit, or None for a one-sided tolerance.
        usl: Upper specification limit, or None.
        cp: Potential capability, or None when the tolerance is one-sided.
        cpu: Capability against the upper limit, or None.
        cpl: Capability against the lower limit, or None.
        cpk: Actual capability, the smaller of cpu and cpl.
        pp: Cp computed with the overall spread.
        ppk: Cpk computed with the overall spread.
    """

    tag: str
    sample_size: int
    subgroup_count: int
    mean: float
    sigma_within: float
    sigma_overall: float
    lsl: float | None
    usl: float | None
    cp: float | None
    cpu: float | None
    cpl: float | None
    cpk: float | None
    pp: float | None
    ppk: float | None

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    @property
    def is_two_sided(self) -> bool:
        """True when the tolerance has both an upper and a lower limit."""
        return self.lsl is not None and self.usl is not None

    @property
    def centring(self) -> float | None:
        """How far off centre the process sits, as a fraction of half the tolerance.

        Zero means perfectly centred. One means the mean is sitting exactly on
        a specification limit. Written k in most references.
        """
        if not self.is_two_sided:
            return None
        assert self.lsl is not None and self.usl is not None
        midpoint = (self.usl + self.lsl) / 2.0
        half_tolerance = (self.usl - self.lsl) / 2.0
        if half_tolerance <= 0.0:
            return None
        return abs(self.mean - midpoint) / half_tolerance

    @property
    def expected_ppm_defective(self) -> float | None:
        """Predicted defects per million parts, assuming a normal process.

        This is a prediction, not a count. It says what the tails of a normal
        curve with this mean and spread would put outside the limits. Real
        processes have heavier tails than a normal curve, so treat it as a
        floor rather than a promise.
        """
        if self.sigma_within <= 0.0:
            return 0.0
        fraction = 0.0
        if self.cpu is not None:
            fraction += normal_tail(3.0 * self.cpu)
        if self.cpl is not None:
            fraction += normal_tail(3.0 * self.cpl)
        return fraction * 1_000_000.0

    @property
    def verdict(self) -> str:
        """A one-word judgement of the Cpk, using the usual industry bands."""
        if self.cpk is None:
            return "NO SPEC"
        if self.cpk < 1.0:
            return "NOT CAPABLE"
        if self.cpk < MINIMUM_ACCEPTABLE_CPK:
            return "MARGINAL"
        if self.cpk < GOOD_CPK:
            return "CAPABLE"
        if self.cpk < EXCELLENT_CPK:
            return "GOOD"
        return "EXCELLENT"

    @property
    def is_acceptable(self) -> bool:
        """True when Cpk meets the usual 1.33 floor."""
        return self.cpk is not None and self.cpk >= MINIMUM_ACCEPTABLE_CPK

    @property
    def centring_loss(self) -> float | None:
        """How much capability is being given away by not being centred.

        Cp minus Cpk. Zero means perfectly centred. A large value means the
        spread is fine and the process simply needs its offset adjusting, which
        is usually the cheapest fix available.
        """
        if self.cp is None or self.cpk is None:
            return None
        return self.cp - self.cpk

    @property
    def stability_gap(self) -> float | None:
        """Cpk minus Ppk. How much capability the drift between subgroups costs.

        Near zero means the process is stable. A large positive value means the
        machine is capable moment to moment but wandering over time.
        """
        if self.cpk is None or self.ppk is None:
            return None
        return self.cpk - self.ppk

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def as_row(self) -> dict[str, object]:
        """Flatten into one dictionary, suitable for a table or a CSV row."""
        return {
            "tag": self.tag,
            "n_parts": self.sample_size,
            "mean": self.mean,
            "sigma_within": self.sigma_within,
            "sigma_overall": self.sigma_overall,
            "cp": self.cp,
            "cpk": self.cpk,
            "pp": self.pp,
            "ppk": self.ppk,
            "ppm": self.expected_ppm_defective,
            "verdict": self.verdict,
        }

    def describe(self) -> str:
        """A short human readable report."""

        def show(value: float | None, places: int = 3) -> str:
            return "n/a" if value is None else f"{value:.{places}f}"

        lines = [
            f"Tag            : {self.tag}",
            f"Parts          : {self.sample_size} "
            f"in {self.subgroup_count} subgroups",
            f"Mean           : {self.mean:.5f}",
            f"Sigma within   : {self.sigma_within:.5f}  (R-bar / d2)",
            f"Sigma overall  : {self.sigma_overall:.5f}  (all individuals)",
            f"Specification  : {show(self.lsl, 3)} to {show(self.usl, 3)}",
            "",
            f"Cp   {show(self.cp)}    potential, ignores centring",
            f"Cpu  {show(self.cpu)}    room to the upper limit",
            f"Cpl  {show(self.cpl)}    room to the lower limit",
            f"Cpk  {show(self.cpk)}    actual, the worse of the two   "
            f"[{self.verdict}]",
            f"Pp   {show(self.pp)}    potential, long-term spread",
            f"Ppk  {show(self.ppk)}    actual, long-term spread",
        ]
        if self.centring is not None:
            lines.append(f"\nCentring k     : {self.centring:.3f}  (0 is centred)")
        if self.centring_loss is not None:
            lines.append(
                f"Cp minus Cpk   : {self.centring_loss:.3f}  "
                "(capability lost to being off centre)"
            )
        if self.stability_gap is not None:
            lines.append(
                f"Cpk minus Ppk  : {self.stability_gap:.3f}  "
                "(capability lost to drift between subgroups)"
            )
        ppm = self.expected_ppm_defective
        if ppm is not None:
            lines.append(f"Predicted      : {ppm:,.1f} defects per million parts")
        return "\n".join(lines)


def _sample_std_dev(values: Sequence[float]) -> float:
    """Standard deviation of a sample, dividing by n minus 1."""
    count = len(values)
    if count < 2:
        return 0.0
    average = sum(values) / count
    variance = sum((v - average) ** 2 for v in values) / (count - 1)
    return math.sqrt(variance)


def capability_from_subgroups(
    subgroups: Sequence[Subgroup], spec: TagSpec
) -> Capability:
    """Compute Cp, Cpk, Pp and Ppk from a set of subgroups.

    Args:
        subgroups: Subgroups from a process believed to be stable. Every one
            must be the same size.
        spec: The tag's specification limits, from the configuration.

    Returns:
        The capability assessment.

    Raises:
        ValueError: if there are no subgroups, the sizes differ, or the tag has
            no specification limits at all.
    """
    if not subgroups:
        raise ValueError("Capability needs at least one subgroup")

    sizes = {group.size for group in subgroups}
    if len(sizes) > 1:
        raise ValueError(
            f"Every subgroup must be the same size, found {sorted(sizes)}"
        )
    if not spec.has_spec_limits:
        raise ValueError(
            f"{spec.name} has no specification limits, so it has no capability. "
            "A counted tag such as ScrapCount is measured with a p-chart instead."
        )

    n = sizes.pop()
    values = [value for group in subgroups for value in group.values]
    mean = sum(values) / len(values)

    mean_range = sum(group.range for group in subgroups) / len(subgroups)
    sigma_within = sigma_from_mean_range(mean_range, n)
    sigma_overall = _sample_std_dev(values)

    def indices(
        sigma: float,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Return (cp, cpu, cpl, cpk) for one estimate of sigma."""
        if sigma <= 0.0:
            # A perfectly repeatable process fits any tolerance infinitely well.
            # Reporting infinity is more honest than dividing by zero.
            infinite = math.inf
            two_sided = spec.is_two_sided
            return (
                infinite if two_sided else None,
                infinite if spec.usl is not None else None,
                infinite if spec.lsl is not None else None,
                infinite,
            )
        cp = None
        if spec.is_two_sided:
            assert spec.usl is not None and spec.lsl is not None
            cp = (spec.usl - spec.lsl) / (6.0 * sigma)
        cpu = None if spec.usl is None else (spec.usl - mean) / (3.0 * sigma)
        cpl = None if spec.lsl is None else (mean - spec.lsl) / (3.0 * sigma)
        one_sided = [v for v in (cpu, cpl) if v is not None]
        cpk = min(one_sided) if one_sided else None
        return cp, cpu, cpl, cpk

    cp, cpu, cpl, cpk = indices(sigma_within)
    pp, _, _, ppk = indices(sigma_overall)

    capability = Capability(
        tag=spec.name,
        sample_size=len(values),
        subgroup_count=len(subgroups),
        mean=mean,
        sigma_within=sigma_within,
        sigma_overall=sigma_overall,
        lsl=spec.lsl,
        usl=spec.usl,
        cp=cp,
        cpu=cpu,
        cpl=cpl,
        cpk=cpk,
        pp=pp,
        ppk=ppk,
    )

    if cpk is not None and cpk < MINIMUM_ACCEPTABLE_CPK:
        logger.info(
            "%s Cpk is %.3f, below the usual %.2f floor (%s)",
            spec.name,
            cpk,
            MINIMUM_ACCEPTABLE_CPK,
            capability.verdict,
        )
    return capability


def rolling_capability(
    subgroups: Sequence[Subgroup], spec: TagSpec, window: int = 20
) -> list[Capability]:
    """Capability over a sliding window, for a Cpk trend line.

    A single Cpk is a snapshot. Watching it fall over a shift is what tells an
    operator that something is going wrong, and it is the trend the dashboard
    plots in Milestone 10.

    Args:
        subgroups: Subgroups in production order.
        spec: The tag's specification limits.
        window: How many subgroups each point is computed from.

    Returns:
        One Capability per window position. Empty when there are fewer
        subgroups than the window.

    Raises:
        ValueError: if the window is smaller than two subgroups.
    """
    if window < 2:
        raise ValueError("A rolling capability window needs at least 2 subgroups")
    if len(subgroups) < window:
        return []
    return [
        capability_from_subgroups(subgroups[start : start + window], spec)
        for start in range(len(subgroups) - window + 1)
    ]


# ---------------------------------------------------------------------------
# Command line demonstration
# ---------------------------------------------------------------------------


def _bore_subgroups(faults, seed: int, count: int):
    """Run the simulator offline and split bore measurements into subgroups."""
    from spc_opcua.config import load_config
    from spc_opcua.simulator.machine import MachineSimulator
    from spc_opcua.spc.subgroups import subgroups_from_values

    config = load_config()
    n = config.subgroup_size
    simulator = MachineSimulator(config, seed=seed, faults=faults)
    values: list[float] = []
    while len(values) < count * n:
        sample = simulator.step()
        if sample.part_completed:
            values.append(sample.values["BoreDiameter"])
    return subgroups_from_values(values, n, tag="BoreDiameter")


def main() -> None:
    """Show what each kind of fault does to Cp and Cpk, and why they differ."""
    from spc_opcua.config import load_config
    from spc_opcua.logging_setup import configure_logging
    from spc_opcua.simulator.faults import (
        FaultSchedule,
        MeanShift,
        ToolWear,
        VarianceInflation,
    )

    configure_logging(level="WARNING")
    spec = load_config().tag("BoreDiameter")

    healthy = _bore_subgroups(FaultSchedule(), seed=1, count=40)
    print("\nHEALTHY PROCESS")
    print("-" * 62)
    print(capability_from_subgroups(healthy, spec).describe())

    scenarios = [
        (
            "MEAN SHIFT of 2 sigma",
            FaultSchedule([MeanShift(tag="BoreDiameter", shift_sigma=2.0)], seed=1),
            "spread unchanged, so Cp holds and only Cpk falls",
        ),
        (
            "VARIANCE INFLATION x2",
            FaultSchedule(
                [VarianceInflation(tag="BoreDiameter", factor=2.0)], seed=1
            ),
            "spread doubled, so BOTH Cp and Cpk fall",
        ),
        (
            "TOOL WEAR",
            FaultSchedule(
                [ToolWear(tag="BoreDiameter", rate_per_hour=-0.05)], seed=1
            ),
            "short-term spread fine, long-term drifting: Cpk stays above Ppk",
        ),
    ]

    print("\n\nWHAT EACH FAULT DOES TO THE INDICES")
    print("-" * 76)
    print(
        f"{'SCENARIO':<24}{'Cp':>7}{'Cpk':>7}{'Pp':>7}{'Ppk':>7}"
        f"{'PPM':>10}  VERDICT"
    )
    print("-" * 76)

    baseline = capability_from_subgroups(healthy, spec)
    rows = [("HEALTHY", baseline, "")] + [
        (name, capability_from_subgroups(_bore_subgroups(f, 1, 40), spec), note)
        for name, f, note in scenarios
    ]
    for name, cap, _ in rows:
        print(
            f"{name:<24}{cap.cp:>7.2f}{cap.cpk:>7.2f}{cap.pp:>7.2f}"
            f"{cap.ppk:>7.2f}{cap.expected_ppm_defective:>10,.0f}  {cap.verdict}"
        )
    print("-" * 76)
    for name, _, note in rows:
        if note:
            print(f"{name:<24}{note}")

    # The Cpk trend, which is what the dashboard will plot.
    wear = FaultSchedule(
        [ToolWear(tag="BoreDiameter", rate_per_hour=-0.05)], seed=1
    )
    trend = rolling_capability(
        _bore_subgroups(wear, seed=1, count=60), spec, window=20
    )
    print("\n\nROLLING Cpk OVER A TOOL-WEAR RUN (window of 20 subgroups)")
    print("-" * 62)
    print(f"{'WINDOW':>7}{'MEAN':>11}{'Cpk':>8}{'Ppk':>8}{'PPM':>12}  VERDICT")
    for i, cap in enumerate(trend):
        if i % 5 == 0 or i == len(trend) - 1:
            print(
                f"{i:>7}{cap.mean:>11.5f}{cap.cpk:>8.2f}{cap.ppk:>8.2f}"
                f"{cap.expected_ppm_defective:>12,.0f}  {cap.verdict}"
            )
    print()


if __name__ == "__main__":
    main()