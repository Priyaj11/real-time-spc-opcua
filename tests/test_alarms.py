"""Tests for turning rule violations into operator alarms."""

from __future__ import annotations

import pytest

from spc_opcua.spc.alarms import (
    CRITICAL,
    WARNING,
    Alarm,
    AlarmLog,
    describe_rule,
    severity_of,
)
from spc_opcua.spc.nelson_rules import Violation


def violation(rule: int, index: int, detail: str = "detail") -> Violation:
    return Violation(
        rule=rule,
        name=f"rule {rule}",
        end_index=index,
        indices=(index,),
        detail=detail,
    )


# --------------------------------------------------------------------------
# Severity
# --------------------------------------------------------------------------


def test_rule_one_is_critical() -> None:
    """Rule 1 means the process is outside its limits right now."""
    assert severity_of(1) == CRITICAL


@pytest.mark.parametrize("rule", [2, 3, 4, 5, 6, 7, 8])
def test_the_pattern_rules_are_warnings(rule: int) -> None:
    """A pattern needs attention, but nothing has left the limits yet."""
    assert severity_of(rule) == WARNING


def test_an_alarm_knows_whether_it_is_critical() -> None:
    log = AlarmLog()
    assert log.record("X-bar", violation(1, 0)).is_critical
    assert not log.record("X-bar", violation(2, 0)).is_critical


# --------------------------------------------------------------------------
# Collapsing repeated firings into one standing alarm
# --------------------------------------------------------------------------


def test_the_first_firing_raises_an_alarm() -> None:
    log = AlarmLog()
    alarm = log.record("X-bar", violation(2, 8))
    assert alarm.first_index == 8
    assert alarm.occurrences == 1
    assert alarm.active


def test_repeated_firings_update_one_alarm_rather_than_stacking() -> None:
    """Rule 2 re-fires on every point of a long run. That is one condition."""
    log = AlarmLog()
    for index in range(8, 20):
        log.record("X-bar", violation(2, index))
    assert len(log.alarms) == 1
    assert log.alarms[0].occurrences == 12
    assert log.alarms[0].first_index == 8
    assert log.alarms[0].last_index == 19


def test_the_same_rule_on_a_different_chart_is_a_different_alarm() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(1, 3))
    log.record("R", violation(1, 3))
    assert len(log.alarms) == 2


def test_different_rules_on_one_chart_are_different_alarms() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(1, 3))
    log.record("X-bar", violation(2, 3))
    assert len(log.alarms) == 2


def test_the_detail_tracks_the_most_recent_firing() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(1, 3, "point at +3.10 sigma"))
    alarm = log.record("X-bar", violation(1, 4, "point at +4.80 sigma"))
    assert alarm.detail == "point at +4.80 sigma"


def test_recording_many_at_once() -> None:
    log = AlarmLog()
    raised = log.record_many("X-bar", [violation(1, 5), violation(2, 5)])
    assert len(raised) == 2


def test_the_duration_spans_first_to_last() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(2, 10))
    alarm = log.record("X-bar", violation(2, 14))
    assert alarm.duration == 5


# --------------------------------------------------------------------------
# Clearing
# --------------------------------------------------------------------------


def test_an_alarm_stays_active_while_the_rule_keeps_firing() -> None:
    log = AlarmLog(clear_after=3)
    log.record("X-bar", violation(2, 10))
    log.expire(11)
    log.record("X-bar", violation(2, 12))
    log.expire(12)
    assert len(log.active) == 1


def test_an_alarm_clears_after_the_quiet_period() -> None:
    log = AlarmLog(clear_after=3)
    log.record("X-bar", violation(2, 10))
    assert log.expire(12) == []  # only two quiet subgroups
    cleared = log.expire(13)
    assert len(cleared) == 1
    assert not log.active
    assert len(log.history) == 1


def test_a_cleared_alarm_stays_in_the_history() -> None:
    log = AlarmLog(clear_after=2)
    log.record("X-bar", violation(2, 5))
    log.expire(10)
    assert len(log.alarms) == 1
    assert not log.alarms[0].active


def test_the_same_rule_firing_again_later_raises_a_new_alarm() -> None:
    """A condition that came back is a new event, not the old one continuing."""
    log = AlarmLog(clear_after=2)
    log.record("X-bar", violation(2, 5))
    log.expire(10)
    log.record("X-bar", violation(2, 40))
    assert len(log.alarms) == 2
    assert len(log.active) == 1


def test_a_clear_period_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        AlarmLog(clear_after=0)


# --------------------------------------------------------------------------
# Ordering and summaries, which is what the screen reads
# --------------------------------------------------------------------------


def test_critical_alarms_sort_above_warnings() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(2, 1))
    log.record("X-bar", violation(1, 20))
    assert log.active[0].rule == 1


def test_within_a_severity_the_most_recent_comes_first() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(2, 1))
    log.record("X-bar", violation(5, 30))
    assert [a.rule for a in log.active] == [5, 2]


def test_the_worst_severity_drives_the_machine_status() -> None:
    log = AlarmLog()
    assert log.worst_severity is None
    log.record("X-bar", violation(2, 1))
    assert log.worst_severity == WARNING
    log.record("X-bar", violation(1, 2))
    assert log.worst_severity == CRITICAL


def test_a_cleared_alarm_stops_driving_the_status() -> None:
    log = AlarmLog(clear_after=1)
    log.record("X-bar", violation(1, 1))
    log.expire(5)
    assert log.worst_severity is None


def test_acknowledging_marks_active_alarms_only() -> None:
    log = AlarmLog(clear_after=1)
    log.record("X-bar", violation(1, 1))
    log.expire(5)  # clears it
    log.record("X-bar", violation(2, 10))
    assert log.acknowledge_all() == 1
    assert log.active[0].acknowledged
    assert not log.history[0].acknowledged


def test_acknowledging_twice_changes_nothing_the_second_time() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(1, 1))
    assert log.acknowledge_all() == 1
    assert log.acknowledge_all() == 0


def test_unacknowledged_criticals_are_reported_separately() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(1, 1))
    log.record("X-bar", violation(2, 1))
    assert len(log.unacknowledged_critical) == 1
    log.acknowledge_all()
    assert log.unacknowledged_critical == ()


def test_the_log_reports_which_rules_it_has_seen() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(1, 1))
    log.record("R", violation(5, 2))
    assert log.rules_seen() == {1, 5}


def test_reset_empties_the_log() -> None:
    log = AlarmLog()
    log.record("X-bar", violation(1, 1))
    log.reset()
    assert log.alarms == ()
    assert log.active == ()


def test_an_empty_log_says_so() -> None:
    assert AlarmLog().summary() == "No alarms."


def test_the_summary_counts_active_and_cleared() -> None:
    log = AlarmLog(clear_after=1)
    log.record("X-bar", violation(1, 1))
    log.expire(5)
    log.record("X-bar", violation(2, 10))
    summary = log.summary()
    assert "1 active" in summary
    assert "1 cleared" in summary


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def test_an_alarm_row_carries_what_a_table_needs() -> None:
    log = AlarmLog()
    row = log.record("X-bar", violation(1, 7)).as_row()
    for key in ("chart", "rule", "severity", "first", "last", "count", "state"):
        assert key in row
    assert row["state"] == "ACTIVE"


def test_a_cleared_alarm_row_says_cleared() -> None:
    log = AlarmLog(clear_after=1)
    log.record("X-bar", violation(1, 1))
    log.expire(5)
    assert log.history[0].as_row()["state"] == "cleared"


def test_an_alarm_prints_readably() -> None:
    log = AlarmLog()
    text = str(log.record("X-bar", violation(1, 7)))
    assert "CRITICAL" in text
    assert "X-bar" in text


def test_a_rule_can_be_described_for_a_tooltip() -> None:
    text = describe_rule(3)
    assert "rising" in text or "falling" in text
    assert "Usually means" in text


def test_describing_an_unknown_rule_does_not_crash() -> None:
    assert "Unknown rule" in describe_rule(99)


def test_the_alarm_key_is_chart_and_rule() -> None:
    alarm = Alarm(
        chart="R",
        rule=5,
        name="x",
        severity=WARNING,
        first_index=0,
        last_index=0,
        occurrences=1,
        detail="",
    )
    assert alarm.key == ("R", 5)