"""Tests for the background live source the dashboard reads from.

The pure-logic tests run everywhere. The ones marked `integration` start a real
OPC UA server in a background thread and let it run for a few seconds, so they
are excluded from the fast loop with `-m "not integration"`.
"""

from __future__ import annotations

import pytest

from spc_opcua.dashboard.live_source import LiveSource, Snapshot


# --------------------------------------------------------------------------
# The snapshot itself, with nothing running
# --------------------------------------------------------------------------


def test_a_fresh_snapshot_is_disconnected_and_empty() -> None:
    """The dashboard draws this before anything has connected."""
    snapshot = Snapshot()
    assert not snapshot.connected
    assert snapshot.parts == 0
    assert snapshot.chart_rows == ()
    assert snapshot.limits is None
    assert not snapshot.has_chart


def test_a_snapshot_knows_it_is_still_baselining() -> None:
    assert Snapshot(phase="baseline").is_baselining
    assert not Snapshot(phase="monitor").is_baselining


def test_a_source_is_not_running_until_it_is_started() -> None:
    source = LiveSource(scenario="none", speed=50.0)
    assert not source.running
    assert not source.snapshot().connected


def test_an_unknown_scenario_is_rejected_at_construction() -> None:
    """Better to fail on the sidebar value than three seconds into a thread."""
    with pytest.raises(ValueError, match="Unknown scenario"):
        LiveSource(scenario="explode")


def test_acknowledging_before_the_engine_exists_is_harmless() -> None:
    assert LiveSource(scenario="none").acknowledge_alarms() == 0


def test_waiting_on_a_condition_that_never_happens_times_out() -> None:
    source = LiveSource(scenario="none")
    assert source.wait_until(lambda s: s.connected, timeout=0.2) is False


# --------------------------------------------------------------------------
# End to end, with a real server and client
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_source_connects_and_produces_subgroups() -> None:
    """Server, client, collector and engine, all through one background thread."""
    source = LiveSource(scenario="none", speed=60.0, baseline_subgroups=5)
    source.start()
    try:
        assert source.wait_until(lambda s: s.connected, timeout=20.0)
        assert source.wait_until(lambda s: s.subgroups >= 3, timeout=30.0)

        snapshot = source.snapshot()
        assert snapshot.parts > 0
        assert snapshot.missed == 0
        assert snapshot.has_chart
        assert snapshot.limits is not None
        assert len(snapshot.chart_rows) == snapshot.subgroups
        assert snapshot.recent_parts
        assert "BoreDiameter" in snapshot.latest
    finally:
        source.stop()

    assert not source.running
    assert not source.snapshot().connected


@pytest.mark.integration
def test_a_worn_tool_eventually_raises_an_alarm() -> None:
    """The whole chain, from a fault in the simulator to a card on the screen."""
    source = LiveSource(scenario="tool-wear", speed=60.0, baseline_subgroups=10)
    source.start()
    try:
        assert source.wait_until(lambda s: s.active_alarms, timeout=90.0)
        snapshot = source.snapshot()
        assert snapshot.spc_status in {"WARNING", "CRITICAL"}
        assert source.acknowledge_alarms() >= 1
        assert all(alarm.acknowledged for alarm in source.snapshot().active_alarms)
    finally:
        source.stop()


@pytest.mark.integration
def test_stopping_a_source_twice_does_not_raise() -> None:
    source = LiveSource(scenario="none", speed=60.0)
    source.start()
    source.wait_until(lambda s: s.connected, timeout=20.0)
    source.stop()
    source.stop()
    assert not source.running