"""The SPC engine: everything from Milestones 7, 8 and 9 behind one door.

Feed it subgroups. It handles the two phases on its own.

    Phase 1, baselining
        The first N subgroups are collected and nothing is judged. You cannot
        monitor against limits you have not calculated yet, and calculating
        them from three subgroups would be worse than useless.

    Phase 2, monitoring
        The limits are frozen. Every new subgroup is plotted on both charts,
        run past the Nelson Rules, folded into a rolling capability window, and
        turned into alarms.

It knows nothing about OPC UA, threads or Streamlit. Subgroups in, an
assessment out, which is what makes it testable without any of those.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from spc_opcua.config import MachineConfig, TagSpec, load_config
from spc_opcua.spc.alarms import Alarm, AlarmLog
from spc_opcua.spc.capability import Capability, capability_from_subgroups
from spc_opcua.spc.control_charts import ChartLimits, ChartPoint, XbarRChart
from spc_opcua.spc.nelson_rules import ALL_RULES, NelsonMonitor, Violation
from spc_opcua.spc.subgroups import Subgroup

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_SUBGROUPS = 25
DEFAULT_CAPABILITY_WINDOW = 20

XBAR_CHART = "X-bar"
R_CHART = "R"


@dataclass(frozen=True)
class EngineUpdate:
    """What one subgroup produced.

    Attributes:
        subgroup: The subgroup that was fed in.
        phase: "baseline" while collecting, "monitor" once limits are frozen.
        point: Where it plotted, or None during baselining.
        violations: Rules that fired on this subgroup.
        raised: Alarms newly raised by this subgroup.
        cleared: Alarms that expired on this subgroup.
        capability: The rolling capability after this subgroup, if there is
            enough data for a window.
    """

    subgroup: Subgroup
    phase: str
    point: ChartPoint | None = None
    violations: tuple[Violation, ...] = ()
    raised: tuple[Alarm, ...] = ()
    cleared: tuple[Alarm, ...] = ()
    capability: Capability | None = None

    @property
    def is_monitoring(self) -> bool:
        """True once the engine has left the baselining phase."""
        return self.phase == "monitor"


class SPCEngine:
    """Control charts, capability and Nelson Rules, driven by one stream of subgroups.

    Example:
        >>> from spc_opcua.spc.subgroups import subgroups_from_values
        >>> engine = SPCEngine(baseline_subgroups=2)
        >>> groups = subgroups_from_values([20.0, 20.01, 19.99, 20.0] * 5, size=4)
        >>> [engine.add(g).phase for g in groups[:3]]
        ['baseline', 'baseline', 'monitor']
    """

    def __init__(
        self,
        config: MachineConfig | None = None,
        chart_tag: str = "BoreDiameter",
        rules: Sequence[int] | None = None,
        baseline_subgroups: int = DEFAULT_BASELINE_SUBGROUPS,
        capability_window: int = DEFAULT_CAPABILITY_WINDOW,
        clear_alarms_after: int = 5,
    ) -> None:
        """Build an engine.

        Args:
            config: The machine definition. Loaded from machine.yaml if omitted.
            chart_tag: Which measurement is charted.
            rules: Which Nelson Rules to apply. All eight if omitted.
            baseline_subgroups: How many subgroups to collect before freezing
                the control limits.
            capability_window: How many subgroups each rolling Cpk uses.
            clear_alarms_after: Quiet subgroups before an alarm clears.

        Raises:
            ValueError: if the baseline is too small to compute limits from.
        """
        if baseline_subgroups < 2:
            raise ValueError("A baseline needs at least 2 subgroups")

        self.config = config if config is not None else load_config()
        self.spec: TagSpec = self.config.tag(chart_tag)
        self.chart_tag = chart_tag
        self.rules = tuple(rules) if rules is not None else ALL_RULES
        self.baseline_subgroups = baseline_subgroups
        self.capability_window = capability_window

        self._baseline: list[Subgroup] = []
        self._monitored: list[Subgroup] = []
        self._chart: XbarRChart | None = None
        self._xbar_monitor = NelsonMonitor(rules=self.rules)
        self._r_monitor = NelsonMonitor(rules=self.rules)
        self._alarms = AlarmLog(clear_after=clear_alarms_after)
        self._capability_trend: list[Capability] = []

    # ------------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------------

    @property
    def is_baselining(self) -> bool:
        """True while still collecting the baseline."""
        return self._chart is None

    @property
    def phase(self) -> str:
        """"baseline" or "monitor"."""
        return "baseline" if self.is_baselining else "monitor"

    @property
    def baseline_progress(self) -> float:
        """How far through the baseline, from 0.0 to 1.0."""
        if not self.is_baselining:
            return 1.0
        return len(self._baseline) / self.baseline_subgroups

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def chart(self) -> XbarRChart | None:
        """The chart, or None while baselining."""
        return self._chart

    @property
    def limits(self) -> ChartLimits | None:
        """The frozen control limits, or None while baselining."""
        return None if self._chart is None else self._chart.limits

    @property
    def points(self) -> tuple[ChartPoint, ...]:
        """Every monitored subgroup as a chart point."""
        return () if self._chart is None else self._chart.points

    @property
    def alarms(self) -> AlarmLog:
        """The alarm log."""
        return self._alarms

    @property
    def capability(self) -> Capability | None:
        """The most recent rolling capability, or None if not enough data."""
        return self._capability_trend[-1] if self._capability_trend else None

    @property
    def capability_trend(self) -> tuple[Capability, ...]:
        """Every rolling capability so far, in order."""
        return tuple(self._capability_trend)

    @property
    def subgroups_monitored(self) -> int:
        """How many subgroups have been plotted."""
        return len(self._monitored)

    @property
    def status(self) -> str:
        """One word for the whole machine, for the top of a dashboard."""
        if self.is_baselining:
            return "BASELINING"
        worst = self._alarms.worst_severity
        if worst is None:
            return "IN CONTROL"
        return worst

    # ------------------------------------------------------------------
    # Feeding
    # ------------------------------------------------------------------

    def add(self, subgroup: Subgroup) -> EngineUpdate:
        """Feed one subgroup and get back everything it produced."""
        if self.is_baselining:
            return self._add_to_baseline(subgroup)
        return self._monitor(subgroup)

    def add_many(self, subgroups: Iterable[Subgroup]) -> list[EngineUpdate]:
        """Feed several subgroups in order."""
        return [self.add(group) for group in subgroups]

    def _add_to_baseline(self, subgroup: Subgroup) -> EngineUpdate:
        """Collect a baseline subgroup, and freeze the limits once there are enough."""
        self._baseline.append(subgroup)
        if len(self._baseline) < self.baseline_subgroups:
            return EngineUpdate(subgroup=subgroup, phase="baseline")

        self._chart = XbarRChart.fit(self._baseline)
        logger.info(
            "Baseline complete after %d subgroups. %s",
            len(self._baseline),
            self._chart.limits.describe().replace("\n", " | "),
        )
        return EngineUpdate(subgroup=subgroup, phase="baseline")

    def _monitor(self, subgroup: Subgroup) -> EngineUpdate:
        """Plot, apply the rules, update capability, and raise alarms."""
        assert self._chart is not None
        self._monitored.append(subgroup)
        point = self._chart.add(subgroup)

        violations = list(self._xbar_monitor.add(point.mean_sigma))
        raised = self._alarms.record_many(XBAR_CHART, violations)

        r_violations = list(self._r_monitor.add(point.range_sigma))
        raised += self._alarms.record_many(R_CHART, r_violations)
        violations += r_violations

        cleared = self._alarms.expire(point.index)

        capability = None
        if len(self._monitored) >= self.capability_window:
            window = self._monitored[-self.capability_window :]
            capability = capability_from_subgroups(window, self.spec)
            self._capability_trend.append(capability)

        return EngineUpdate(
            subgroup=subgroup,
            phase="monitor",
            point=point,
            violations=tuple(violations),
            raised=tuple(raised),
            cleared=tuple(cleared),
            capability=capability,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def chart_frame(self):
        """Every plotted point as a pandas table, for the dashboard."""
        import pandas as pd

        if self._chart is None:
            return pd.DataFrame(
                columns=["subgroup", "mean", "range", "mean_sigma", "range_sigma"]
            )
        return self._chart.to_frame()

    def capability_frame(self):
        """The rolling capability trend as a pandas table."""
        import pandas as pd

        if not self._capability_trend:
            return pd.DataFrame(columns=["cp", "cpk", "pp", "ppk", "ppm", "verdict"])
        rows = []
        for offset, capability in enumerate(self._capability_trend):
            row = capability.as_row()
            row["subgroup"] = offset + self.capability_window - 1
            rows.append(row)
        return pd.DataFrame(rows)

    def alarm_frame(self, include_cleared: bool = True):
        """The alarm log as a pandas table."""
        import pandas as pd

        chosen = self._alarms.alarms if include_cleared else self._alarms.active
        if not chosen:
            return pd.DataFrame(
                columns=["chart", "rule", "name", "severity", "first", "last", "count"]
            )
        return pd.DataFrame([alarm.as_row() for alarm in chosen])

    def summary(self) -> str:
        """A short human readable report of the whole engine."""
        lines = [f"Status         : {self.status}", f"Phase          : {self.phase}"]
        if self.is_baselining:
            lines.append(
                f"Baseline       : {len(self._baseline)}"
                f"/{self.baseline_subgroups} subgroups"
            )
            return "\n".join(lines)
        assert self._chart is not None
        lines += [
            f"Monitored      : {self.subgroups_monitored} subgroups",
            f"R chart in control: {self._chart.range_is_in_control}",
            "",
            self._chart.limits.describe(),
        ]
        if self.capability is not None:
            lines += [
                "",
                f"Cpk            : {self.capability.cpk:.3f} "
                f"[{self.capability.verdict}]",
                f"Ppk            : {self.capability.ppk:.3f}",
            ]
        lines += ["", self._alarms.summary()]
        return "\n".join(lines)