"""Tests for the data collection pipeline."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from spc_opcua.config import MachineConfig, load_config
from spc_opcua.opcua_client import ClientSettings, MachineClient, TagUpdate
from spc_opcua.opcua_server import MachineServer, ServerSettings
from spc_opcua.pipeline import DataCollector, collect_from_client

START = datetime(2026, 8, 26, 9, 0, 0)


@pytest.fixture
def config() -> MachineConfig:
    return load_config()


def make_collector(config: MachineConfig, **kwargs) -> DataCollector:
    """A collector fed from a synthetic stream, so there is no opening snapshot."""
    kwargs.setdefault("skip_initial_snapshot", False)
    return DataCollector(config, **kwargs)


def update(tag: str, value: float | int | str, seconds: float = 0.0) -> TagUpdate:
    """A TagUpdate as the client would deliver it."""
    return TagUpdate(
        tag=tag,
        value=value,
        source_timestamp=START + timedelta(seconds=seconds),
        received_at=seconds,
    )


def feed_parts(collector: DataCollector, values: list[float]) -> None:
    """Feed one bore measurement per part, with PartCount advancing alongside."""
    for i, value in enumerate(values, start=1):
        collector.handle(update("PartCount", i, seconds=i * 12.0))
        collector.handle(update("BoreDiameter", value, seconds=i * 12.0))


# --------------------------------------------------------------------------
# Recognising a part
# --------------------------------------------------------------------------


def test_one_bore_measurement_is_one_part(config: MachineConfig) -> None:
    """A subscription only fires on change, and the bore is held between parts."""
    collector = make_collector(config)
    collector.handle(update("BoreDiameter", 20.001))
    collector.handle(update("BoreDiameter", 19.998))
    assert len(collector.parts) == 2


def test_other_tags_do_not_create_parts(config: MachineConfig) -> None:
    collector = make_collector(config)
    for tag in ("Torque", "Temperature", "Vibration", "CycleTime", "Status"):
        collector.handle(update(tag, 1.0))
    assert collector.parts == ()
    assert collector.updates_received == 5


def test_context_tags_are_snapshotted_onto_the_part(config: MachineConfig) -> None:
    collector = make_collector(config)
    collector.handle(update("Torque", 46.2))
    collector.handle(update("Temperature", 41.0))
    collector.handle(update("Status", "RUNNING"))
    collector.handle(update("BoreDiameter", 20.004))

    part = collector.parts[0]
    assert part.context["Torque"] == pytest.approx(46.2)
    assert part.context["Temperature"] == pytest.approx(41.0)
    assert part.context["Status"] == "RUNNING"


def test_a_part_records_the_machines_own_part_number(config: MachineConfig) -> None:
    collector = make_collector(config)
    collector.handle(update("PartCount", 17))
    collector.handle(update("BoreDiameter", 20.0))
    assert collector.parts[0].part_index == 17


def test_the_part_carries_the_server_timestamp(config: MachineConfig) -> None:
    collector = make_collector(config)
    collector.handle(update("BoreDiameter", 20.0, seconds=36.0))
    assert collector.parts[0].timestamp == START + timedelta(seconds=36.0)


def test_parts_are_numbered_in_arrival_order(config: MachineConfig) -> None:
    collector = make_collector(config)
    feed_parts(collector, [20.0, 20.1, 20.2])
    assert [p.sequence for p in collector.parts] == [0, 1, 2]


# --------------------------------------------------------------------------
# Forming subgroups
# --------------------------------------------------------------------------


def test_a_subgroup_appears_every_five_parts(config: MachineConfig) -> None:
    collector = make_collector(config)
    assert collector.subgroup_size == 5
    feed_parts(collector, [20.0] * 4)
    assert collector.subgroups == ()
    feed_parts(collector, [20.0])
    assert len(collector.subgroups) == 1


def test_handle_returns_the_subgroup_it_completed(config: MachineConfig) -> None:
    collector = make_collector(config)
    results = [collector.handle(update("BoreDiameter", 20.0)) for _ in range(5)]
    assert results[:4] == [None, None, None, None]
    assert results[4] is not None
    assert results[4].size == 5


def test_handle_many_returns_every_completed_subgroup(config: MachineConfig) -> None:
    collector = make_collector(config)
    updates = [update("BoreDiameter", 20.0 + i * 0.001) for i in range(12)]
    completed = collector.handle_many(updates)
    assert len(completed) == 2
    assert collector.pending_parts == 2


def test_the_subgroup_statistics_come_from_the_parts(config: MachineConfig) -> None:
    collector = make_collector(config)
    feed_parts(collector, [20.00, 20.02, 19.98, 20.01, 19.99])
    group = collector.subgroups[0]
    assert group.mean == pytest.approx(20.0)
    assert group.range == pytest.approx(0.04)
    assert group.part_indices == (1, 2, 3, 4, 5)


def test_a_custom_subgroup_size_overrides_the_config(config: MachineConfig) -> None:
    collector = make_collector(config, subgroup_size=3)
    feed_parts(collector, [20.0] * 6)
    assert len(collector.subgroups) == 2


# --------------------------------------------------------------------------
# Warm-up
# --------------------------------------------------------------------------


def test_warmup_parts_are_discarded_before_collecting(
    config: MachineConfig,
) -> None:
    """A spindle warming up is not a stable process, so those parts are dropped."""
    collector = make_collector(config, warmup_parts=10)
    feed_parts(collector, [20.0] * 15)
    assert collector.parts_skipped_during_warmup == 10
    assert len(collector.parts) == 5
    assert len(collector.subgroups) == 1


def test_no_warmup_by_default(config: MachineConfig) -> None:
    collector = make_collector(config)
    feed_parts(collector, [20.0] * 5)
    assert collector.parts_skipped_during_warmup == 0
    assert len(collector.parts) == 5


# --------------------------------------------------------------------------
# Data integrity
# --------------------------------------------------------------------------


def test_nothing_is_missing_when_every_part_arrives(config: MachineConfig) -> None:
    collector = make_collector(config)
    feed_parts(collector, [20.0] * 20)
    assert collector.machine_part_count == 20
    assert collector.missed_parts == 0


def test_a_dropped_measurement_is_detected(config: MachineConfig) -> None:
    """The machine says 20 parts, we have 18, so two went missing."""
    collector = make_collector(config)
    feed_parts(collector, [20.0] * 18)
    collector.handle(update("PartCount", 20))
    assert collector.missed_parts == 2


def test_warmup_parts_do_not_count_as_missing(config: MachineConfig) -> None:
    collector = make_collector(config, warmup_parts=5)
    feed_parts(collector, [20.0] * 15)
    assert collector.missed_parts == 0


def test_missed_parts_is_zero_before_the_machine_reports_anything(
    config: MachineConfig,
) -> None:
    collector = make_collector(config)
    collector.handle(update("BoreDiameter", 20.0))
    assert collector.machine_part_count is None
    assert collector.missed_parts == 0


# --------------------------------------------------------------------------
# Stoppages
# --------------------------------------------------------------------------


def test_a_stoppage_discards_the_partial_subgroup(config: MachineConfig) -> None:
    collector = make_collector(config)
    feed_parts(collector, [20.0, 20.0, 20.0])
    collector.mark_stoppage()
    assert collector.pending_parts == 0
    feed_parts(collector, [20.0] * 5)
    assert len(collector.subgroups) == 1


# --------------------------------------------------------------------------
# Tables and files
# --------------------------------------------------------------------------


def test_the_part_table_has_one_row_per_part(config: MachineConfig) -> None:
    collector = make_collector(config)
    feed_parts(collector, [20.0, 20.1, 20.2])
    frame = collector.parts_frame()
    assert len(frame) == 3
    assert "value" in frame.columns
    assert "part_index" in frame.columns


def test_the_subgroup_table_has_one_row_per_subgroup(
    config: MachineConfig,
) -> None:
    collector = make_collector(config)
    feed_parts(collector, [20.0] * 15)
    frame = collector.subgroups_frame()
    assert len(frame) == 3
    assert set(["mean", "range", "min", "max"]).issubset(frame.columns)


def test_empty_tables_still_have_columns(config: MachineConfig) -> None:
    """A dashboard drawing an empty chart must not crash on a missing column."""
    collector = make_collector(config)
    assert list(collector.parts_frame().columns)
    assert list(collector.subgroups_frame().columns)


def test_csv_files_are_written_and_readable(
    config: MachineConfig, tmp_path: Path
) -> None:
    import pandas as pd

    collector = make_collector(config)
    feed_parts(collector, [20.0 + i * 0.001 for i in range(10)])

    parts_path = collector.write_parts_csv(tmp_path / "nested" / "parts.csv")
    groups_path = collector.write_subgroups_csv(tmp_path / "nested" / "groups.csv")

    assert parts_path.exists() and groups_path.exists()
    assert len(pd.read_csv(parts_path)) == 10
    assert len(pd.read_csv(groups_path)) == 2


def test_the_summary_reports_the_key_numbers(config: MachineConfig) -> None:
    collector = make_collector(config)
    feed_parts(collector, [20.0] * 10)
    summary = collector.summary()
    assert "Parts recorded : 10" in summary
    assert "BoreDiameter" in summary


# --------------------------------------------------------------------------
# Configuration errors
# --------------------------------------------------------------------------


def test_an_unknown_chart_tag_fails_at_construction(config: MachineConfig) -> None:
    """Better to fail now than to collect nothing and report an empty chart."""
    with pytest.raises(KeyError):
        DataCollector(config, chart_tag="NoSuchTag")


def test_a_different_chart_tag_can_be_collected(config: MachineConfig) -> None:
    collector = make_collector(config, chart_tag="Torque")
    for value in (44.0, 45.0, 46.0, 45.5, 44.5):
        collector.handle(update("Torque", value))
    assert len(collector.subgroups) == 1
    assert collector.subgroups[0].tag == "Torque"


# --------------------------------------------------------------------------
# End to end, over a real socket
# --------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def running_server() -> AsyncIterator[MachineServer]:
    server = MachineServer(
        settings=ServerSettings(port=free_port(), speed_factor=100.0)
    )
    await server.start()
    publisher = asyncio.create_task(server.run())
    try:
        yield server
    finally:
        publisher.cancel()
        try:
            await publisher
        except asyncio.CancelledError:
            pass
        await server.stop()


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
async def test_subgroups_form_from_a_live_opcua_stream(
    running_server: MachineServer,
) -> None:
    collector = DataCollector()
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe()
        completed = await collect_from_client(
            c, collector, duration_s=25.0, subgroup_limit=3
        )

    assert len(completed) == 3
    assert all(group.size == 5 for group in completed)
    assert len(collector.parts) >= 15
    assert collector.missed_parts == 0

    spec = load_config().tag("BoreDiameter")
    for group in completed:
        assert abs(group.mean - spec.nominal) < 10 * spec.std_dev
        assert group.range > 0.0


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
async def test_collecting_stops_when_the_stream_goes_quiet(
    running_server: MachineServer,
) -> None:
    """No subscription means no traffic, and the collector must not hang."""
    collector = DataCollector()
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        completed = await collect_from_client(c, collector, duration_s=0.5)
    assert completed == []


# --------------------------------------------------------------------------
# The subscription's opening snapshot
# --------------------------------------------------------------------------


def test_the_opening_snapshot_is_discarded_by_default(
    config: MachineConfig,
) -> None:
    """Subscribing delivers the CURRENT value at once. That is not a new part."""
    collector = DataCollector(config)
    collector.handle(update("BoreDiameter", 20.000))
    assert collector.parts == ()
    assert collector.initial_snapshot_skipped

    collector.handle(update("BoreDiameter", 20.001))
    assert len(collector.parts) == 1


def test_discarding_the_snapshot_can_be_switched_off(
    config: MachineConfig,
) -> None:
    collector = DataCollector(config, skip_initial_snapshot=False)
    collector.handle(update("BoreDiameter", 20.000))
    assert len(collector.parts) == 1


def test_the_first_real_part_carries_full_context(config: MachineConfig) -> None:
    """Without the skip, part one would arrive before any other tag had spoken."""
    collector = DataCollector(config)
    collector.handle(update("BoreDiameter", 20.000))  # opening snapshot
    collector.handle(update("Torque", 45.0))
    collector.handle(update("PartCount", 1))
    collector.handle(update("BoreDiameter", 20.002))

    part = collector.parts[0]
    assert part.part_index == 1
    assert part.context["Torque"] == pytest.approx(45.0)