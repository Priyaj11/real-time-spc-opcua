"""Run every scenario many times and measure what actually happens.

One replicate is a complete production run: a healthy baseline, limits frozen,
then the fault, monitored to the end of the window. Nothing here is estimated
or assumed; every number in the results comes from counting things that
happened in a simulated run.

What is measured, and what each one means
-----------------------------------------
detected
    Did any alarm fire during the monitored window? On a faulted scenario that
    is a true positive. On the healthy scenario it is a false alarm.

detection_subgroups / detection_parts
    How long the fault ran before the first alarm. Counted from the first
    monitored subgroup, so a fault caught immediately scores 1.

first_scrap_part
    The first part whose TRUE dimension is outside the customer's tolerance.
    Truth, not reading, so a drifting gauge never invents scrap.

warning_parts
    Parts made between the alarm and the first scrap part. Positive means the
    system warned before anything was actually wrong, which is the entire
    argument for control charts over inspection. Negative means the first bad
    part was already made when the alarm arrived.

scrap_avoided
    A stated counterfactual, not a measurement: how many of the window's scrap
    parts came after the alarm, and so would not have been made if the line had
    stopped when the alarm fired. Whether a real plant stops on a warning is a
    management decision, which is why this is reported as a separate column and
    not folded into anything else.

Replicates
----------
Every scenario is run many times with different machine seeds. A single run
tells you almost nothing: detection latency is a random variable, and one run
of one seed is one sample from it. The median and the spread are what mean
something, and the healthy scenario's false alarm rate needs many runs before
it is worth quoting at all.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from spc_opcua.config import MachineConfig, TagSpec, load_config
from spc_opcua.evaluation.scenarios import (
    BASELINE_SUBGROUPS,
    MONITOR_SUBGROUPS,
    SCENARIOS,
    TAG,
    Scenario,
)
from spc_opcua.simulator.offline import part_readings_and_truth
from spc_opcua.spc.engine import SPCEngine
from spc_opcua.spc.nelson_rules import COMMON_RULES
from spc_opcua.spc.subgroups import subgroups_from_values

logger = logging.getLogger(__name__)

DEFAULT_REPLICATES = 30

# Seeds are consecutive from here, so a result can always be reproduced by
# quoting the scenario name and the replicate number.
FIRST_SEED = 1000


@dataclass(frozen=True)
class RunResult:
    """What one complete production run produced.

    Every field is a count or an index taken from that run. None means the
    thing never happened: no alarm, or no scrap.
    """

    scenario: str
    kind: str
    seed: int

    detected: bool
    detection_subgroups: int | None
    detection_parts: int | None
    first_rule: int | None
    critical: bool

    # Every monitored subgroup, and how many of them raised something. On the
    # healthy scenario these two give the per-subgroup false alarm rate, which
    # is the only false alarm number that does not depend on how long you
    # happened to watch for.
    monitored_subgroups: int
    alarm_subgroups: int

    scrap_parts: int
    first_scrap_part: int | None
    scrap_after_alarm: int
    warning_parts: int | None

    final_cpk: float | None

    def as_row(self) -> dict[str, Any]:
        """Flatten for a table."""
        return asdict(self)


def _out_of_spec(value: float, spec: TagSpec) -> bool:
    """True when a part's true dimension is outside the customer's tolerance."""
    return (spec.lsl is not None and value < spec.lsl) or (
        spec.usl is not None and value > spec.usl
    )


def run_once(
    scenario: Scenario,
    seed: int,
    config: MachineConfig | None = None,
    rules: tuple[int, ...] = COMMON_RULES,
    baseline: int = BASELINE_SUBGROUPS,
    monitor: int = MONITOR_SUBGROUPS,
) -> RunResult:
    """Run one scenario once, from a healthy baseline to the end of the window.

    Args:
        scenario: Which fault to inject.
        seed: Seed for the machine's noise. Different seeds are different runs
            of the same machine, not different machines.
        config: Machine definition. Loaded from machine.yaml if omitted.
        rules: Which Nelson Rules to apply.
        baseline: Subgroups collected before the limits are frozen.
        monitor: Subgroups monitored after the fault begins.

    Returns:
        Everything measured about that run.
    """
    config = config if config is not None else load_config()
    size = config.subgroup_size
    spec = config.tag(TAG)
    total_parts = (baseline + monitor) * size

    readings, truth = part_readings_and_truth(
        scenario.faults, seed=seed, parts=total_parts, tag=TAG
    )
    subgroups = subgroups_from_values(list(readings), size, tag=TAG)

    engine = SPCEngine(
        config,
        rules=rules,
        baseline_subgroups=baseline,
        capability_window=20,
    )

    detection_subgroups: int | None = None
    first_rule: int | None = None
    critical = False
    alarm_subgroups = 0

    for position, subgroup in enumerate(subgroups):
        update = engine.add(subgroup)
        if not update.raised:
            continue
        alarm_subgroups += 1
        if detection_subgroups is None:
            # position counts every subgroup; the monitored ones start at
            # `baseline`, and a fault caught on the very first one scores 1.
            detection_subgroups = position - baseline + 1
            first_rule = update.raised[0].rule
            critical = any(alarm.is_critical for alarm in update.raised)

    detected = detection_subgroups is not None
    detection_parts = detection_subgroups * size if detected else None

    # Scrap is counted from the true dimension, and only over the monitored
    # window. A sensor fault leaves truth untouched, so it can never score here.
    monitored_truth = truth[baseline * size :]
    scrap_indices = [
        i
        for i, value in enumerate(monitored_truth, start=1)
        if _out_of_spec(value, spec)
    ]
    first_scrap_part = scrap_indices[0] if scrap_indices else None

    scrap_after_alarm = 0
    warning_parts: int | None = None
    if detected and detection_parts is not None:
        scrap_after_alarm = sum(1 for i in scrap_indices if i > detection_parts)
        if first_scrap_part is not None:
            warning_parts = first_scrap_part - detection_parts

    capability = engine.capability
    return RunResult(
        scenario=scenario.name,
        kind=scenario.kind,
        seed=seed,
        detected=detected,
        detection_subgroups=detection_subgroups,
        detection_parts=detection_parts,
        first_rule=first_rule,
        critical=critical,
        monitored_subgroups=monitor,
        alarm_subgroups=alarm_subgroups,
        scrap_parts=len(scrap_indices),
        first_scrap_part=first_scrap_part,
        scrap_after_alarm=scrap_after_alarm,
        warning_parts=warning_parts,
        final_cpk=None if capability is None else capability.cpk,
    )


def run_scenario(
    scenario: Scenario,
    replicates: int = DEFAULT_REPLICATES,
    config: MachineConfig | None = None,
    **kwargs: Any,
) -> list[RunResult]:
    """Run one scenario several times with consecutive seeds."""
    config = config if config is not None else load_config()
    results = [
        run_once(scenario, seed=FIRST_SEED + i, config=config, **kwargs)
        for i in range(replicates)
    ]
    caught = sum(r.detected for r in results)
    logger.info("%-18s %d/%d detected", scenario.name, caught, replicates)
    return results


def run_all(
    replicates: int = DEFAULT_REPLICATES,
    scenarios: tuple[Scenario, ...] = SCENARIOS,
    config: MachineConfig | None = None,
    **kwargs: Any,
) -> list[RunResult]:
    """Run every scenario. This is the whole experiment."""
    config = config if config is not None else load_config()
    results: list[RunResult] = []
    for item in scenarios:
        results.extend(
            run_scenario(item, replicates=replicates, config=config, **kwargs)
        )
    return results


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSummary:
    """One scenario's results, reduced to the numbers worth quoting.

    Latency is summarised by the median rather than the mean, because a run
    that never detects has no latency at all and a mean over only the
    successful runs would flatter a detector that misses half the time.
    """

    scenario: str
    kind: str
    severity: str
    replicates: int
    detected: int
    detection_rate: float
    median_subgroups: float | None
    median_parts: float | None
    worst_subgroups: int | None
    median_warning_parts: float | None
    scrap_parts: int
    scrap_after_alarm: int
    monitored_subgroups: int
    alarm_subgroups: int
    rules_that_fired: tuple[int, ...]

    @property
    def scrap_avoidable(self) -> float | None:
        """Share of scrap made after the alarm, if the line stopped on it."""
        if self.scrap_parts == 0:
            return None
        return self.scrap_after_alarm / self.scrap_parts

    @property
    def alarm_rate_per_subgroup(self) -> float:
        """Share of monitored subgroups that raised something.

        On the healthy scenario this IS the false alarm rate, and it is the
        only version of that number that does not depend on how long you
        happened to watch. A rate of five per cent sounds small and means an
        operator sees a false alarm roughly every twenty subgroups.
        """
        if not self.monitored_subgroups:
            return 0.0
        return self.alarm_subgroups / self.monitored_subgroups

    @property
    def subgroups_between_alarms(self) -> float | None:
        """Average run length: subgroups between one false alarm and the next.

        Derived as one divided by the measured per-subgroup rate, not measured
        directly. Quoting it that way avoids censoring: a run that happens not
        to alarm inside the window would otherwise be counted as if it never
        would.
        """
        rate = self.alarm_rate_per_subgroup
        return None if rate == 0 else 1.0 / rate

    def as_row(self) -> dict[str, Any]:
        """Flatten for a table, with the derived share included."""
        row = asdict(self)
        row["rules_that_fired"] = ",".join(str(r) for r in self.rules_that_fired)
        row["scrap_avoidable"] = self.scrap_avoidable
        row["alarm_rate_per_subgroup"] = self.alarm_rate_per_subgroup
        row["subgroups_between_alarms"] = self.subgroups_between_alarms
        return row


def summarise(scenario: Scenario, results: list[RunResult]) -> ScenarioSummary:
    """Reduce one scenario's replicates to a single row."""
    caught = [r for r in results if r.detected]
    latencies = [r.detection_subgroups for r in caught if r.detection_subgroups]
    warnings = [r.warning_parts for r in caught if r.warning_parts is not None]
    rules = sorted({r.first_rule for r in caught if r.first_rule is not None})

    return ScenarioSummary(
        scenario=scenario.name,
        kind=scenario.kind,
        severity=scenario.severity,
        replicates=len(results),
        detected=len(caught),
        detection_rate=len(caught) / len(results) if results else 0.0,
        median_subgroups=median(latencies) if latencies else None,
        median_parts=median(latencies) * 5 if latencies else None,
        worst_subgroups=max(latencies) if latencies else None,
        median_warning_parts=median(warnings) if warnings else None,
        scrap_parts=sum(r.scrap_parts for r in results),
        scrap_after_alarm=sum(r.scrap_after_alarm for r in results),
        monitored_subgroups=sum(r.monitored_subgroups for r in results),
        alarm_subgroups=sum(r.alarm_subgroups for r in results),
        rules_that_fired=tuple(rules),
    )


def summarise_all(
    results: list[RunResult], scenarios: tuple[Scenario, ...] = SCENARIOS
) -> list[ScenarioSummary]:
    """One summary row per scenario, in catalogue order."""
    by_name: dict[str, list[RunResult]] = {}
    for result in results:
        by_name.setdefault(result.scenario, []).append(result)
    return [
        summarise(item, by_name[item.name]) for item in scenarios if item.name in by_name
    ]


def headline(summaries: list[ScenarioSummary]) -> dict[str, Any]:
    """The three or four numbers that belong in a README.

    Detection rate is averaged over the faulted scenarios only, because a
    healthy run has nothing to detect. The false alarm rate comes from the
    healthy scenario alone, because that is the only place an alarm is
    unambiguously wrong.
    """
    faulted = [s for s in summaries if s.kind != "healthy"]
    healthy = [s for s in summaries if s.kind == "healthy"]

    caught = sum(s.detected for s in faulted)
    runs = sum(s.replicates for s in faulted)
    latencies = [s.median_subgroups for s in faulted if s.median_subgroups is not None]
    warnings = [
        s.median_warning_parts for s in faulted if s.median_warning_parts is not None
    ]
    scrap = sum(s.scrap_parts for s in faulted)
    avoidable = sum(s.scrap_after_alarm for s in faulted)

    return {
        "faulted_runs": runs,
        "faulted_detected": caught,
        "detection_rate": caught / runs if runs else 0.0,
        "median_detection_subgroups": median(latencies) if latencies else None,
        "median_warning_parts": median(warnings) if warnings else None,
        "scrap_parts": scrap,
        "scrap_after_alarm": avoidable,
        "scrap_avoidable_share": avoidable / scrap if scrap else None,
        "healthy_runs": sum(s.replicates for s in healthy),
        "healthy_false_alarms": sum(s.detected for s in healthy),
        # Two different false alarm numbers, both honest, easily confused.
        # The window rate answers "will an operator see a false alarm during a
        # shift"; the per-subgroup rate answers "how often is a plotted point
        # wrongly flagged". The first depends on how long you watch. The
        # second does not, and is the one to quote.
        "false_alarm_rate_per_window": (healthy[0].detection_rate if healthy else None),
        "false_alarm_rate_per_subgroup": (
            healthy[0].alarm_rate_per_subgroup if healthy else None
        ),
        "subgroups_between_false_alarms": (
            healthy[0].subgroups_between_alarms if healthy else None
        ),
    }