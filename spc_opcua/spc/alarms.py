"""Turning rule violations into alarms an operator can actually use.

A rule violation and an alarm are not the same thing, and conflating them is
how you build a dashboard nobody looks at.

A violation is a fact about one point. Rule 2 fires when nine points sit on one
side of the centre line, and it keeps firing on every point after that while
the run continues, because the run is still nine long. Ten consecutive points
below the centre produce two violations, twenty produce twelve. Showing all of
those as separate alarms buries the operator.

An alarm is a condition. It is raised once when a rule starts firing, it stays
ACTIVE while the rule keeps firing, it records how many times it has fired, and
it CLEARS after the rule has been quiet for a while. That is what a real human
machine interface does, and it is why this module exists between the Nelson
Rules and the screen.

Severity
--------
    CRITICAL   the process is outside its control limits right now (rule 1)
    WARNING    a non-random pattern is present but nothing has left the limits

Both need attention. Only one needs it this minute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Iterable

from spc_opcua.spc.nelson_rules import RULES, Violation

logger = logging.getLogger(__name__)

CRITICAL = "CRITICAL"
WARNING = "WARNING"

# Rules that mean the process is outside its limits right now.
CRITICAL_RULES = frozenset({1})

# How many quiet subgroups before an active alarm clears itself. Long enough
# that a rule firing intermittently stays as one alarm rather than flapping.
DEFAULT_CLEAR_AFTER = 5


@dataclass(frozen=True)
class Alarm:
    """One condition, raised once and updated as it persists.

    Attributes:
        chart: Which chart raised it, "X-bar" or "R".
        rule: Which Nelson Rule.
        name: Short name of the rule.
        severity: CRITICAL or WARNING.
        first_index: Subgroup where the condition was first detected.
        last_index: Subgroup where it most recently fired.
        occurrences: How many times the rule has fired while this alarm stood.
        detail: The most recent description from the rule.
        active: True while the rule is still firing or recently was.
        acknowledged: True once an operator has pressed acknowledge.
    """

    chart: str
    rule: int
    name: str
    severity: str
    first_index: int
    last_index: int
    occurrences: int
    detail: str
    active: bool = True
    acknowledged: bool = False

    @property
    def key(self) -> tuple[str, int]:
        """What makes two firings the same alarm: same chart, same rule."""
        return (self.chart, self.rule)

    @property
    def is_critical(self) -> bool:
        """True when the process is outside its control limits."""
        return self.severity == CRITICAL

    @property
    def duration(self) -> int:
        """How many subgroups the condition has spanned."""
        return self.last_index - self.first_index + 1

    def as_row(self) -> dict[str, object]:
        """Flatten into one dictionary, suitable for a table."""
        return {
            "chart": self.chart,
            "rule": self.rule,
            "name": self.name,
            "severity": self.severity,
            "first": self.first_index,
            "last": self.last_index,
            "count": self.occurrences,
            "state": "ACTIVE" if self.active else "cleared",
            "ack": "yes" if self.acknowledged else "",
            "detail": self.detail,
        }

    def __str__(self) -> str:
        state = "ACTIVE" if self.active else "cleared"
        return (
            f"[{self.severity}] {self.chart} rule {self.rule} ({self.name}) "
            f"subgroups {self.first_index}-{self.last_index}, "
            f"{self.occurrences}x, {state}"
        )


def severity_of(rule: int) -> str:
    """CRITICAL when the rule means the limits have been broken, else WARNING."""
    return CRITICAL if rule in CRITICAL_RULES else WARNING


class AlarmLog:
    """Collapses a stream of rule violations into raised, standing and cleared alarms.

    Example:
        >>> from spc_opcua.spc.nelson_rules import Violation
        >>> log = AlarmLog()
        >>> v = Violation(1, "beyond 3 sigma", 4, (4,), "point at +3.50 sigma")
        >>> alarm = log.record("X-bar", v)
        >>> alarm.severity
        'CRITICAL'
        >>> log.record("X-bar", v).occurrences
        2
    """

    def __init__(self, clear_after: int = DEFAULT_CLEAR_AFTER) -> None:
        """Build an alarm log.

        Args:
            clear_after: How many subgroups a rule must stay quiet before its
                alarm clears.
        """
        if clear_after < 1:
            raise ValueError("clear_after must be at least 1 subgroup")
        self.clear_after = clear_after
        self._alarms: list[Alarm] = []
        self._open: dict[tuple[str, int], int] = {}  # key to index in _alarms

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alarms(self) -> tuple[Alarm, ...]:
        """Every alarm ever raised, oldest first."""
        return tuple(self._alarms)

    @property
    def active(self) -> tuple[Alarm, ...]:
        """Alarms that are still standing, most recent first."""
        return tuple(
            sorted(
                (a for a in self._alarms if a.active),
                key=lambda a: (a.severity != CRITICAL, -a.last_index),
            )
        )

    @property
    def history(self) -> tuple[Alarm, ...]:
        """Alarms that have cleared, most recent first."""
        return tuple(
            sorted((a for a in self._alarms if not a.active), key=lambda a: -a.last_index)
        )

    @property
    def unacknowledged_critical(self) -> tuple[Alarm, ...]:
        """Active critical alarms nobody has acknowledged yet."""
        return tuple(a for a in self.active if a.is_critical and not a.acknowledged)

    @property
    def worst_severity(self) -> str | None:
        """CRITICAL if any active alarm is critical, WARNING if any, else None."""
        active = self.active
        if not active:
            return None
        return CRITICAL if any(a.is_critical for a in active) else WARNING

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, chart: str, violation: Violation) -> Alarm:
        """Take one rule violation and raise or update the matching alarm.

        Args:
            chart: "X-bar" or "R".
            violation: What the rule reported.

        Returns:
            The alarm this violation belongs to, new or updated.
        """
        key = (chart, violation.rule)
        position = self._open.get(key)

        if position is None:
            alarm = Alarm(
                chart=chart,
                rule=violation.rule,
                name=violation.name,
                severity=severity_of(violation.rule),
                first_index=violation.end_index,
                last_index=violation.end_index,
                occurrences=1,
                detail=violation.detail,
            )
            self._alarms.append(alarm)
            self._open[key] = len(self._alarms) - 1
            logger.info("Alarm raised: %s", alarm)
            return alarm

        standing = self._alarms[position]
        updated = replace(
            standing,
            last_index=violation.end_index,
            occurrences=standing.occurrences + 1,
            detail=violation.detail,
        )
        self._alarms[position] = updated
        return updated

    def record_many(self, chart: str, violations: Iterable[Violation]) -> list[Alarm]:
        """Record several violations from the same chart."""
        return [self.record(chart, violation) for violation in violations]

    def expire(self, current_index: int) -> list[Alarm]:
        """Clear any active alarm whose rule has been quiet long enough.

        Args:
            current_index: The subgroup just processed.

        Returns:
            The alarms that cleared on this call.
        """
        cleared: list[Alarm] = []
        for key, position in list(self._open.items()):
            standing = self._alarms[position]
            if current_index - standing.last_index < self.clear_after:
                continue
            self._alarms[position] = replace(standing, active=False)
            del self._open[key]
            cleared.append(self._alarms[position])
            logger.info("Alarm cleared: %s", self._alarms[position])
        return cleared

    def acknowledge_all(self) -> int:
        """Mark every active alarm as seen by an operator.

        Returns:
            How many alarms were acknowledged.
        """
        count = 0
        for position, alarm in enumerate(self._alarms):
            if alarm.active and not alarm.acknowledged:
                self._alarms[position] = replace(alarm, acknowledged=True)
                count += 1
        return count

    def reset(self) -> None:
        """Forget every alarm."""
        self._alarms.clear()
        self._open.clear()

    def rules_seen(self) -> set[int]:
        """Which Nelson Rules have raised an alarm at any point."""
        return {a.rule for a in self._alarms}

    def summary(self) -> str:
        """A short human readable report."""
        if not self._alarms:
            return "No alarms."
        lines = [
            f"{len(self.active)} active, {len(self.history)} cleared, "
            f"{len(self._alarms)} total"
        ]
        for alarm in self.active:
            lines.append(f"  {alarm}")
        return "\n".join(lines)


def describe_rule(rule: int) -> str:
    """One line explaining what a rule means, for an alarm tooltip."""
    spec = RULES.get(rule)
    if spec is None:
        return f"Unknown rule {rule}"
    return f"{spec.detects}. Usually means: {spec.matters}."