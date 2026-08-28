"""The eight Nelson Rules.

A control limit alone only catches a point that has already gone three sigma
out. By then the process has usually been misbehaving for a while. The Nelson
Rules, published by Lloyd Nelson in 1984, add seven more patterns that each
mean "this is no longer random", and most of them fire long before anything
crosses a limit.

They are written in terms of ZONES, which is why Milestone 7 built them:

    Zone C   within 1 sigma of the centre line
    Zone B   between 1 and 2 sigma
    Zone A   between 2 and 3 sigma
    beyond   outside the control limits

Every rule in this module takes a list of SIGMA DISTANCES, not raw
measurements. A sigma distance is (value - centre) / sigma, so +2.4 means two
point four sigma above the centre line. Working in those units means each rule
is a handful of comparisons on plain numbers, testable without a chart, a
machine or a network anywhere in sight.

The catch nobody mentions
-------------------------
Each rule has its own false alarm rate, and they add up. Rule 1 on its own
fires on about 0.27 percent of points from a healthy process. Turn on all eight
and the combined rate is several times that. More rules means earlier
detection AND more crying wolf, and an operator who stops believing the alarms
has a worse system than one with a single rule. Choosing which rules to enable
is a real engineering decision, which is why this module lets you pick.

Which formulation
-----------------
Nelson's wording is sometimes ambiguous and different textbooks implement small
variations. Where that happens, the choice made here is written in the rule's
docstring, so a reviewer can see it rather than guess.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The longest pattern any rule looks at. A streaming monitor only ever needs
# this many recent points in memory.
MAX_RULE_WINDOW = 15


@dataclass(frozen=True)
class Violation:
    """One rule firing at one point.

    Attributes:
        rule: Which Nelson Rule fired, 1 to 8.
        name: Short name of the rule.
        end_index: The point at which the pattern completed, which is the
            moment it became detectable.
        indices: Every point involved in the pattern.
        detail: One line describing what was seen.
    """

    rule: int
    name: str
    end_index: int
    indices: tuple[int, ...]
    detail: str

    @property
    def start_index(self) -> int:
        """The first point involved in the pattern."""
        return self.indices[0]

    @property
    def span(self) -> int:
        """How many points the pattern covers."""
        return len(self.indices)

    def __str__(self) -> str:
        return f"Rule {self.rule} ({self.name}) at point {self.end_index}: {self.detail}"


# ---------------------------------------------------------------------------
# The eight rules
# ---------------------------------------------------------------------------


def rule_1(z: Sequence[float]) -> list[Violation]:
    """One point beyond three sigma.

    Detects: a single wild value, or a shift so large it lands outside the
    limits immediately.
    Matters because: this is the classic out-of-control signal, and the only
    one most people know. On a healthy normal process it fires about 0.27
    percent of the time.
    Triggering data: [0, 0, 0, 3.5]
    """
    return [
        Violation(
            rule=1,
            name="beyond 3 sigma",
            end_index=i,
            indices=(i,),
            detail=f"point at {value:+.2f} sigma",
        )
        for i, value in enumerate(z)
        if abs(value) > 3.0
    ]


def rule_2(z: Sequence[float], run: int = 9) -> list[Violation]:
    """Nine points in a row on the same side of the centre line.

    Detects: a sustained shift in the process average, too small to break a
    control limit.
    Matters because: nine coin flips all landing the same way is about a one in
    256 event. A process that keeps making parts slightly oversize is drifting
    towards scrap even though no single point looks alarming.
    Triggering data: nine consecutive positive values, for example [0.5] * 9
    Note: a point sitting exactly on the centre line breaks the run, since it
    is on neither side.

    Args:
        z: Sigma distances of the plotted points, in order.
        run: How many in a row. Nine is Nelson's number.
    """
    return _same_sign_run(z, run, rule=2, name="9 on one side")


def rule_3(z: Sequence[float], run: int = 6) -> list[Violation]:
    """Six points in a row, all rising or all falling.

    Detects: a trend. Tool wear, a gauge drifting out of calibration, a machine
    warming up.
    Matters because: this is the rule that catches tool wear early, before the
    drift is large enough to leave the control limits.
    Triggering data: [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5], or the reverse.
    Deliberately all inside one sigma, so this data trips rule 3 and
    nothing else: a trend is detectable long before it reaches a limit.
    Note: strictly monotonic. Two equal values break the run, which is the
    common formulation and avoids a flat line counting as a trend.

    Args:
        z: Sigma distances of the plotted points, in order.
        run: How many in a row. Six is Nelson's number.
    """
    violations: list[Violation] = []
    if len(z) < run:
        return violations
    for end in range(run - 1, len(z)):
        window = z[end - run + 1 : end + 1]
        steps = [later - earlier for earlier, later in zip(window, window[1:], strict=False)]
        if all(step > 0 for step in steps):
            direction = "rising"
        elif all(step < 0 for step in steps):
            direction = "falling"
        else:
            continue
        violations.append(
            Violation(
                rule=3,
                name="6 point trend",
                end_index=end,
                indices=tuple(range(end - run + 1, end + 1)),
                detail=f"{run} points steadily {direction}",
            )
        )
    return violations


def rule_4(z: Sequence[float], run: int = 14) -> list[Violation]:
    """Fourteen points in a row alternating up and down.

    Detects: systematic oscillation. Two machines, two fixtures, two operators
    or two material lots feeding the same chart alternately, or an over-eager
    operator adjusting after every part.
    Matters because: over-adjustment is one of the most common ways a well
    meaning operator makes a process worse. Deming called it tampering.
    Triggering data: [0, 1, 0, 1, ...] for fourteen points
    Args:
        z: Sigma distances of the plotted points, in order.
        run: How many in a row. Fourteen is Nelson's number.
    """
    violations: list[Violation] = []
    if len(z) < run:
        return violations
    for end in range(run - 1, len(z)):
        window = z[end - run + 1 : end + 1]
        steps = [later - earlier for earlier, later in zip(window, window[1:], strict=False)]
        if any(step == 0 for step in steps):
            continue
        if all((a > 0) != (b > 0) for a, b in zip(steps, steps[1:], strict=False)):
            violations.append(
                Violation(
                    rule=4,
                    name="14 alternating",
                    end_index=end,
                    indices=tuple(range(end - run + 1, end + 1)),
                    detail=f"{run} points alternating up and down",
                )
            )
    return violations


def rule_5(z: Sequence[float]) -> list[Violation]:
    """Two out of three consecutive points beyond two sigma, same side.

    Detects: a moderate shift, roughly two sigma, that has not yet produced a
    point outside the limits.
    Matters because: it reacts far faster than rule 2 to a shift of that size.
    Triggering data: [2.5, 0.1, 2.4]
    Note: the third point may be anywhere. Both extreme points must be on the
    same side, since one high and one low is spread, not a shift.
    """
    return _m_of_n_beyond(z, m=2, n=3, threshold=2.0, rule=5, name="2 of 3 beyond 2s")


def rule_6(z: Sequence[float]) -> list[Violation]:
    """Four out of five consecutive points beyond one sigma, same side.

    Detects: a smaller sustained shift, around one sigma.
    Matters because: it sits between rule 5 and rule 2 in sensitivity, catching
    shifts too small for rule 5 and faster than waiting for nine in a row.
    Triggering data: [1.2, 1.5, 0.2, 1.1, 1.3]
    """
    return _m_of_n_beyond(z, m=4, n=5, threshold=1.0, rule=6, name="4 of 5 beyond 1s")


def rule_7(z: Sequence[float], run: int = 15) -> list[Violation]:
    """Fifteen points in a row all within one sigma of the centre.

    Detects: data that is TOO well behaved. This one surprises people, because
    it fires when everything looks perfect.
    Matters because: it almost never means the process got better. It usually
    means the control limits are too wide, because the baseline was taken while
    something unusual was happening, or the measurement system has lost
    resolution, or somebody is quietly rounding the numbers. It is also the
    rule that catches a stuck sensor reporting the same value forever.
    Triggering data: [0.2, 0.1, -0.1, -0.2, 0.3] repeated three times.
    Fifteen values all inside one sigma, arranged so no run and no
    alternation trips another rule at the same time.
    """
    violations: list[Violation] = []
    if len(z) < run:
        return violations
    for end in range(run - 1, len(z)):
        window = z[end - run + 1 : end + 1]
        if all(abs(value) < 1.0 for value in window):
            violations.append(
                Violation(
                    rule=7,
                    name="15 hugging centre",
                    end_index=end,
                    indices=tuple(range(end - run + 1, end + 1)),
                    detail=f"{run} points all inside 1 sigma, limits may be too wide",
                )
            )
    return violations


def rule_8(z: Sequence[float], run: int = 8) -> list[Violation]:
    """Eight points in a row beyond one sigma, appearing on both sides.

    Detects: a process avoiding its own centre line. Usually two distinct
    streams mixed onto one chart, for example two spindles, two cavities of a
    mould, or two suppliers of the same raw material.
    Matters because: the average looks fine, the spread looks fine, and the
    process is actually two different processes wearing a trenchcoat.
    Triggering data: [1.5, -1.5, 1.6, -1.4, 1.5, -1.6, 1.4, -1.5]
    Note: Nelson's wording is "on both sides of the centre line with none in
    Zone C". Requiring both sides is what keeps this rule distinct from rule 6,
    which already catches a one-sided run beyond one sigma.
    """
    violations: list[Violation] = []
    if len(z) < run:
        return violations
    for end in range(run - 1, len(z)):
        window = z[end - run + 1 : end + 1]
        if not all(abs(value) > 1.0 for value in window):
            continue
        if not (any(v > 0 for v in window) and any(v < 0 for v in window)):
            continue
        violations.append(
            Violation(
                rule=8,
                name="8 avoiding centre",
                end_index=end,
                indices=tuple(range(end - run + 1, end + 1)),
                detail=f"{run} points outside 1 sigma on both sides, possible mixture",
            )
        )
    return violations


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------


def _same_sign_run(
    z: Sequence[float], run: int, rule: int, name: str
) -> list[Violation]:
    """Find every run of `run` consecutive points on the same side of centre."""
    violations: list[Violation] = []
    if len(z) < run:
        return violations
    for end in range(run - 1, len(z)):
        window = z[end - run + 1 : end + 1]
        if all(value > 0 for value in window):
            side = "above"
        elif all(value < 0 for value in window):
            side = "below"
        else:
            continue
        violations.append(
            Violation(
                rule=rule,
                name=name,
                end_index=end,
                indices=tuple(range(end - run + 1, end + 1)),
                detail=f"{run} consecutive points {side} the centre line",
            )
        )
    return violations


def _m_of_n_beyond(
    z: Sequence[float], m: int, n: int, threshold: float, rule: int, name: str
) -> list[Violation]:
    """Find every window of n where at least m points are beyond a threshold, same side.

    The m extreme points must all be on the same side. One high and one low is
    a spread problem, not a shift, and the R chart is what catches that.
    """
    violations: list[Violation] = []
    if len(z) < n:
        return violations
    for end in range(n - 1, len(z)):
        start = end - n + 1
        window = z[start : end + 1]
        for sign, label in ((1, "above"), (-1, "below")):
            hits = [
                start + offset
                for offset, value in enumerate(window)
                if sign * value > threshold
            ]
            # The pattern must be detectable AT this point, so the newest point
            # has to be one of the extreme ones. Otherwise the same window
            # would report again on every later point it slides through.
            if len(hits) >= m and end in hits:
                violations.append(
                    Violation(
                        rule=rule,
                        name=name,
                        end_index=end,
                        indices=tuple(range(start, end + 1)),
                        detail=(
                            f"{len(hits)} of {n} points beyond {threshold:.0f} sigma "
                            f"{label} the centre line"
                        ),
                    )
                )
                break
    return violations


# ---------------------------------------------------------------------------
# The rule registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleSpec:
    """One rule, with everything needed to describe and run it.

    Attributes:
        number: 1 to 8.
        name: Short name.
        detects: What pattern it looks for.
        matters: Why anyone should care.
        window: How many points the pattern spans.
        check: The function that finds violations in a list of sigma distances.
    """

    number: int
    name: str
    detects: str
    matters: str
    window: int
    check: Callable[[Sequence[float]], list[Violation]]


RULES: dict[int, RuleSpec] = {
    1: RuleSpec(
        1,
        "beyond 3 sigma",
        "one point outside the control limits",
        "a single wild value or a large sudden shift",
        1,
        rule_1,
    ),
    2: RuleSpec(
        2,
        "9 on one side",
        "nine points in a row on the same side of the centre",
        "a sustained shift too small to break a limit",
        9,
        rule_2,
    ),
    3: RuleSpec(
        3,
        "6 point trend",
        "six points in a row all rising or all falling",
        "tool wear, gauge drift, a machine warming up",
        6,
        rule_3,
    ),
    4: RuleSpec(
        4,
        "14 alternating",
        "fourteen points in a row alternating up and down",
        "over-adjustment, or two streams alternating on one chart",
        14,
        rule_4,
    ),
    5: RuleSpec(
        5,
        "2 of 3 beyond 2s",
        "two of three consecutive points beyond two sigma, same side",
        "a moderate shift, faster than waiting for nine in a row",
        3,
        rule_5,
    ),
    6: RuleSpec(
        6,
        "4 of 5 beyond 1s",
        "four of five consecutive points beyond one sigma, same side",
        "a smaller sustained shift",
        5,
        rule_6,
    ),
    7: RuleSpec(
        7,
        "15 hugging centre",
        "fifteen points in a row all within one sigma",
        "limits too wide, lost measurement resolution, or a stuck sensor",
        15,
        rule_7,
    ),
    8: RuleSpec(
        8,
        "8 avoiding centre",
        "eight points in a row beyond one sigma on both sides",
        "two different processes mixed onto one chart",
        8,
        rule_8,
    ),
}

ALL_RULES: tuple[int, ...] = tuple(sorted(RULES))

# A common conservative default. Rules 1, 2, 3, 5 and 6 catch shifts and
# trends; 4, 7 and 8 catch rarer structural problems and contribute most of the
# extra false alarms. Many shops run only these five.
COMMON_RULES: tuple[int, ...] = (1, 2, 3, 5, 6)


def apply_rules(
    z: Sequence[float], rules: Iterable[int] | None = None
) -> list[Violation]:
    """Run the chosen rules over a whole series of sigma distances.

    Args:
        z: Sigma distances, in order. Use ControlLimits.sigma_distance to build
            them from raw subgroup means or ranges.
        rules: Which rule numbers to apply. All eight if omitted.

    Returns:
        Every violation found, ordered by where the pattern completed.

    Raises:
        KeyError: if a rule number does not exist.
    """
    chosen = ALL_RULES if rules is None else tuple(rules)
    found: list[Violation] = []
    for number in chosen:
        try:
            spec = RULES[number]
        except KeyError as exc:
            raise KeyError(
                f"There is no Nelson Rule {number}. Rules are {ALL_RULES}."
            ) from exc
        found.extend(spec.check(z))
    found.sort(key=lambda v: (v.end_index, v.rule))
    return found


class NelsonMonitor:
    """Applies the Nelson Rules to a stream of points, one at a time.

    Only the most recent MAX_RULE_WINDOW points are kept, because no rule looks
    further back than that. Each point is reported at most once per rule, at the
    moment its pattern completes.

    Example:
        >>> monitor = NelsonMonitor(rules=[1])
        >>> monitor.add(0.5)
        []
        >>> [v.rule for v in monitor.add(4.0)]
        [1]
    """

    def __init__(self, rules: Iterable[int] | None = None) -> None:
        """Build a monitor.

        Args:
            rules: Which rule numbers to apply. All eight if omitted.
        """
        self.rules: tuple[int, ...] = ALL_RULES if rules is None else tuple(rules)
        for number in self.rules:
            if number not in RULES:
                raise KeyError(
                    f"There is no Nelson Rule {number}. Rules are {ALL_RULES}."
                )
        self._window: deque[float] = deque(maxlen=MAX_RULE_WINDOW)
        self._count = 0
        self._violations: list[Violation] = []

    @property
    def points_seen(self) -> int:
        """How many points have been fed in."""
        return self._count

    @property
    def violations(self) -> tuple[Violation, ...]:
        """Every violation raised so far, in order."""
        return tuple(self._violations)

    @property
    def first_violation(self) -> Violation | None:
        """The earliest violation, or None if nothing has fired."""
        return self._violations[0] if self._violations else None

    def rules_fired(self) -> set[int]:
        """Which rule numbers have fired at least once."""
        return {v.rule for v in self._violations}

    def reset(self) -> None:
        """Forget every point and every violation."""
        self._window.clear()
        self._count = 0
        self._violations.clear()

    def add(self, sigma_distance: float) -> list[Violation]:
        """Feed one point and return any rule that fired on it.

        Args:
            sigma_distance: (value - centre) / sigma for the new point.

        Returns:
            Violations whose pattern completes at this point.
        """
        self._window.append(float(sigma_distance))
        self._count += 1
        newest = len(self._window) - 1
        offset = self._count - 1 - newest  # absolute index of window position 0

        window = list(self._window)
        fired: list[Violation] = []
        for number in self.rules:
            for violation in RULES[number].check(window):
                if violation.end_index != newest:
                    continue
                absolute = Violation(
                    rule=violation.rule,
                    name=violation.name,
                    end_index=violation.end_index + offset,
                    indices=tuple(i + offset for i in violation.indices),
                    detail=violation.detail,
                )
                fired.append(absolute)
                logger.info("%s", absolute)

        self._violations.extend(fired)
        return fired

    def add_many(self, values: Iterable[float]) -> list[Violation]:
        """Feed many points in order and return everything that fired."""
        fired: list[Violation] = []
        for value in values:
            fired.extend(self.add(value))
        return fired


# ---------------------------------------------------------------------------
# Command line demonstration
# ---------------------------------------------------------------------------


# One hand-built series per rule, chosen so that each trips its own rule and
# nothing else. These are the same series the tests use.
TRIGGERING_DATA: dict[int, list[float]] = {
    1: [0.0, 0.0, 0.0, 3.5],
    2: [0.5] * 9,
    3: [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5],
    4: [0.0, 1.0] * 7,
    5: [2.5, 0.1, 2.4],
    6: [1.2, 1.5, 0.2, 1.1, 1.3],
    7: [0.2, 0.1, -0.1, -0.2, 0.3] * 3,
    8: [1.5, -1.5, 1.6, -1.4, 1.5, -1.6, 1.4, -1.5],
}


def _bore_sigmas(faults, seed: int, count: int, baseline_seed: int = 1):
    """Fit limits on a healthy baseline, then return sigma distances for a run."""
    from spc_opcua.config import load_config
    from spc_opcua.simulator.faults import FaultSchedule
    from spc_opcua.simulator.machine import MachineSimulator
    from spc_opcua.spc.control_charts import XbarRChart
    from spc_opcua.spc.subgroups import subgroups_from_values

    config = load_config()
    n = config.subgroup_size

    def groups(schedule, seed_value: int, how_many: int):
        simulator = MachineSimulator(config, seed=seed_value, faults=schedule)
        values: list[float] = []
        while len(values) < how_many * n:
            sample = simulator.step()
            if sample.part_completed:
                values.append(sample.values["BoreDiameter"])
        return subgroups_from_values(values, n, tag="BoreDiameter")

    chart = XbarRChart.fit(groups(FaultSchedule(), baseline_seed, 25))
    points = chart.add_many(groups(faults, seed, count))
    return [p.mean_sigma for p in points]


def main() -> None:
    """Show each rule firing, then compare detection and false alarms."""
    from spc_opcua.logging_setup import configure_logging
    from spc_opcua.simulator.faults import FaultSchedule, ToolWear

    configure_logging(level="WARNING")

    print("\nTHE EIGHT RULES, EACH ON DATA BUILT TO TRIP IT")
    print("-" * 78)
    print(f"{'#':>2}  {'NAME':<20}{'SPAN':>5}  DETECTS")
    print("-" * 78)
    for number in ALL_RULES:
        spec = RULES[number]
        data = TRIGGERING_DATA[number]
        fired = sorted({v.rule for v in apply_rules(data)})
        mark = "ok" if fired == [number] else f"also {fired}"
        print(f"{spec.number:>2}  {spec.name:<20}{spec.window:>5}  {spec.detects}")
        print(f"    {'':<20}{'':>5}  matters: {spec.matters}")
        print(
            f"    {'':<20}{'':>5}  data: "
            f"{data if len(data) <= 9 else str(data[:6]) + ' ...'}  [{mark}]"
        )
    print("-" * 78)

    # Detection: rule 1 alone versus all eight, on a tool-wear run.
    wear = FaultSchedule(
        [ToolWear(tag="BoreDiameter", start_s=0.0, rate_per_hour=-0.05)], seed=1
    )
    sigmas = _bore_sigmas(wear, seed=1, count=40)

    print("\n\nTOOL WEAR: WHEN DOES EACH SET OF RULES FIRST FIRE?")
    print("-" * 78)
    for label, chosen in (
        ("Rule 1 only", (1,)),
        ("Common five (1,2,3,5,6)", COMMON_RULES),
        ("All eight", ALL_RULES),
    ):
        monitor = NelsonMonitor(rules=chosen)
        monitor.add_many(sigmas)
        first = monitor.first_violation
        where = "never" if first is None else f"subgroup {first.end_index}"
        which = "" if first is None else f"  (rule {first.rule}, {first.name})"
        print(f"{label:<26}{where:<16}{which}")

    monitor = NelsonMonitor()
    monitor.add_many(sigmas)
    print(f"\nRules that fired at all: {sorted(monitor.rules_fired())}")
    print("\nFirst eight violations in order:")
    for violation in monitor.violations[:8]:
        print(f"  {violation}")

    # False alarms: the honest cost of turning more rules on.
    print("\n\nFALSE ALARM RATE ON A HEALTHY PROCESS")
    print("-" * 78)
    healthy_runs = [
        _bore_sigmas(FaultSchedule(), seed=s, count=40) for s in range(50, 70)
    ]
    total_points = sum(len(run) for run in healthy_runs)
    print(f"{'RULE SET':<26}{'ALARMS':>8}{'POINTS':>9}{'RATE':>9}")
    for label, chosen in (
        ("Rule 1 only", (1,)),
        ("Common five (1,2,3,5,6)", COMMON_RULES),
        ("All eight", ALL_RULES),
    ):
        alarms = 0
        for run in healthy_runs:
            monitor = NelsonMonitor(rules=chosen)
            monitor.add_many(run)
            # Count POINTS that raised at least one alarm, not total violations.
            # A run of nine re-fires on every later point in the run, so counting
            # violations would flatter rule 1 and punish rule 2 unfairly.
            alarms += len({v.end_index for v in monitor.violations})
        print(
            f"{label:<26}{alarms:>8}{total_points:>9}"
            f"{alarms / total_points * 100:>8.2f}%"
        )
    print(
        "\nMore rules detect faster and cry wolf more often. Which set to enable\n"
        "is an engineering decision, not a default."
    )
    print()


if __name__ == "__main__":
    main()