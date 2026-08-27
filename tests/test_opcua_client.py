"""Integration tests for the OPC UA client.

A real server, a real socket, a real subscription. Marked `integration` so the
fast unit tests can still be run alone with:

    python -m pytest -m "not integration"
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from spc_opcua.config import load_config
from spc_opcua.opcua_client import (
    DEFAULT_TAGS,
    ClientSettings,
    MachineClient,
    TagUpdate,
)
from spc_opcua.opcua_server import MachineServer, ServerSettings

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

CONTINUOUS_TAGS = ("Torque", "Temperature", "Vibration")
PER_PART_TAGS = ("BoreDiameter", "CycleTime")


def free_port() -> int:
    """Ask the operating system for a port nobody else is using."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def running_server() -> AsyncIterator[MachineServer]:
    """One server, publishing continuously at 20x, for the whole module."""
    server = MachineServer(settings=ServerSettings(port=free_port(), speed_factor=20.0))
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


@pytest_asyncio.fixture(loop_scope="module")
async def client(running_server: MachineServer) -> AsyncIterator[MachineClient]:
    """A connected client, disconnected again afterwards."""
    machine_client = MachineClient(ClientSettings(endpoint=running_server.endpoint))
    await machine_client.connect()
    try:
        yield machine_client
    finally:
        await machine_client.disconnect()


# --------------------------------------------------------------------------
# Connecting
# --------------------------------------------------------------------------


async def test_the_client_connects_and_reports_it(client: MachineClient) -> None:
    assert client.connected


async def test_the_namespace_is_resolved_from_the_uri(
    client: MachineClient, running_server: MachineServer
) -> None:
    assert client.namespace_index == running_server.namespace_index


async def test_using_the_client_before_connecting_is_an_error() -> None:
    machine_client = MachineClient(ClientSettings(endpoint="opc.tcp://127.0.0.1:1/x/"))
    with pytest.raises(RuntimeError, match="connect"):
        await machine_client.read_all()
    with pytest.raises(RuntimeError, match="not connected"):
        _ = machine_client.namespace_index


async def test_connecting_to_nothing_fails_rather_than_hanging() -> None:
    settings = ClientSettings(
        endpoint=f"opc.tcp://127.0.0.1:{free_port()}/bore01/server/",
        connect_timeout_s=2.0,
    )
    machine_client = MachineClient(settings)
    with pytest.raises((OSError, asyncio.TimeoutError, ConnectionError)):
        await machine_client.connect()
    assert not machine_client.connected


async def test_disconnecting_twice_is_harmless(
    running_server: MachineServer,
) -> None:
    machine_client = MachineClient(ClientSettings(endpoint=running_server.endpoint))
    await machine_client.connect()
    await machine_client.disconnect()
    await machine_client.disconnect()
    assert not machine_client.connected


async def test_connecting_twice_is_harmless(client: MachineClient) -> None:
    await client.connect()
    assert client.connected


# --------------------------------------------------------------------------
# Reading, the polling way
# --------------------------------------------------------------------------


async def test_read_all_returns_every_followed_tag(client: MachineClient) -> None:
    values = await client.read_all()
    assert set(values) == set(DEFAULT_TAGS)


async def test_reading_one_tag_gives_a_plausible_value(
    client: MachineClient,
) -> None:
    spec = load_config().tag("BoreDiameter")
    value = await client.read("BoreDiameter")
    assert isinstance(value, float)
    assert abs(value - spec.nominal) < 20 * spec.std_dev


async def test_reading_an_unfollowed_tag_names_the_followed_ones(
    client: MachineClient,
) -> None:
    with pytest.raises(KeyError, match="Torque"):
        await client.read("NoSuchTag")


async def test_a_client_can_follow_a_subset_of_tags(
    running_server: MachineServer,
) -> None:
    machine_client = MachineClient(
        ClientSettings(endpoint=running_server.endpoint),
        tags=("Torque", "BoreDiameter"),
    )
    async with machine_client:
        values = await machine_client.read_all()
    assert set(values) == {"Torque", "BoreDiameter"}


# --------------------------------------------------------------------------
# Subscribing, the push way
# --------------------------------------------------------------------------


async def test_subscribing_returns_one_handle_per_tag(
    client: MachineClient,
) -> None:
    handles = await client.subscribe()
    assert len(handles) == len(DEFAULT_TAGS)


async def test_updates_arrive_without_being_asked_for(
    running_server: MachineServer,
) -> None:
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe()
        updates = await c.collect(1.0)
    assert len(updates) > 20
    assert all(isinstance(u, TagUpdate) for u in updates)


async def test_updates_carry_the_servers_own_timestamp(
    running_server: MachineServer,
) -> None:
    """Order by the server's clock, not by when we happened to receive it."""
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe(["Torque"])
        updates = await c.collect(1.0, limit=5)
    assert updates
    assert all(u.source_timestamp is not None for u in updates)


async def test_the_latest_snapshot_fills_in_as_updates_arrive(
    running_server: MachineServer,
) -> None:
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        assert c.latest == {}
        await c.subscribe()
        await c.collect(1.0)
        assert set(CONTINUOUS_TAGS) <= set(c.latest)


async def test_continuous_tags_update_far_more_often_than_per_part_tags(
    running_server: MachineServer,
) -> None:
    """Torque changes every 100 ms; a bore is measured once every 12 s."""
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe()
        updates = await c.collect(2.0)

    counts: dict[str, int] = {}
    for update in updates:
        counts[update.tag] = counts.get(update.tag, 0) + 1

    continuous = min(counts.get(name, 0) for name in CONTINUOUS_TAGS)
    per_part = max(counts.get(name, 0) for name in PER_PART_TAGS)
    assert continuous > 5 * max(per_part, 1)


async def test_a_subscription_only_fires_on_change(
    running_server: MachineServer,
) -> None:
    """Status stays RUNNING, so it notifies once and then goes quiet."""
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe(["Status"])
        updates = await c.collect(1.5)
    assert len(updates) <= 3
    assert all(u.tag == "Status" for u in updates)


async def test_subscribing_to_a_subset_ignores_the_rest(
    running_server: MachineServer,
) -> None:
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe(["Torque"])
        updates = await c.collect(1.0, limit=20)
    assert updates
    assert {u.tag for u in updates} == {"Torque"}


async def test_the_update_counter_matches_what_was_delivered(
    running_server: MachineServer,
) -> None:
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe(["Torque"])
        updates = await c.collect(1.0)
        assert c.update_count >= len(updates)


async def test_updates_can_be_iterated_with_a_limit(
    running_server: MachineServer,
) -> None:
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe(["Torque"])
        seen = [u async for u in c.updates(limit=7, timeout_s=5.0)]
    assert len(seen) == 7


async def test_iterating_gives_up_when_nothing_arrives(
    running_server: MachineServer,
) -> None:
    """No subscription means no traffic, and the iterator must not hang."""
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        seen = [u async for u in c.updates(limit=5, timeout_s=0.3)]
    assert seen == []


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------


async def test_age_is_none_for_a_tag_never_heard_from(
    running_server: MachineServer,
) -> None:
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        assert c.age_s("Torque") is None


async def test_age_is_small_right_after_an_update(
    running_server: MachineServer,
) -> None:
    async with MachineClient(ClientSettings(endpoint=running_server.endpoint)) as c:
        await c.subscribe(["Torque"])
        await c.collect(1.0, limit=5)
        age = c.age_s("Torque")
    assert age is not None
    assert age < 1.0


# --------------------------------------------------------------------------
# Back pressure
# --------------------------------------------------------------------------


async def test_a_full_buffer_drops_the_oldest_rather_than_blocking(
    running_server: MachineServer,
) -> None:
    """A slow consumer must never make the server's connection stall."""
    settings = ClientSettings(endpoint=running_server.endpoint, max_queue=25)
    async with MachineClient(settings) as c:
        await c.subscribe()
        await asyncio.sleep(1.5)  # let updates pile up while nobody drains
        assert c.update_count > 25
        assert c.dropped_count > 0


# --------------------------------------------------------------------------
# The record type
# --------------------------------------------------------------------------


async def test_numeric_and_text_updates_are_distinguishable() -> None:
    numeric = TagUpdate("Torque", 45.0, None, 0.0)
    text = TagUpdate("Status", "RUNNING", None, 0.0)
    assert numeric.is_numeric
    assert not text.is_numeric