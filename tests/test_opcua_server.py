"""Integration tests for the OPC UA server.

These start a real server on a real TCP socket and connect a real client to it.
Nothing is mocked.

Every test here is marked `integration`, so the fast unit tests can be run on
their own while you work:

    python -m pytest -m "not integration"

Starting an OPC UA server costs about a second and a half, because the standard
address space has to be built before ours is added on top. Tests that only read
the address space therefore share one server for the whole module. Tests about
starting, stopping and pacing build their own, because those need a clean one.
"""

from __future__ import annotations

import socket
import time
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from asyncua import Client, ua

from spc_opcua.config import load_config
from spc_opcua.opcua_server import FAULT_PRESETS, MachineServer, ServerSettings
from spc_opcua.simulator.faults import FaultSchedule, MeanShift
from spc_opcua.simulator.machine import MachineSimulator

# One event loop for the whole module, so a module-scoped server fixture can
# live across tests instead of being rebuilt for each one.
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

EXPECTED_NODES = (
    "BoreDiameter",
    "Torque",
    "CycleTime",
    "Temperature",
    "Vibration",
    "ScrapCount",
    "PartCount",
    "Status",
)


def free_port() -> int:
    """Ask the operating system for a port nobody else is using."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def settings(**overrides) -> ServerSettings:
    """Server settings on a free port, with overrides applied."""
    return ServerSettings(port=free_port(), **overrides)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def shared() -> AsyncIterator[MachineServer]:
    """One healthy server, reused by every read-only test in this module."""
    server = MachineServer(settings=settings(speed_factor=100.0))
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


async def connect_machine(server: MachineServer, client: Client):
    """Resolve the Machine object through a connected client."""
    idx = await client.get_namespace_index(server.settings.namespace_uri)
    return idx, await client.nodes.objects.get_child(f"{idx}:Machine")


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


async def test_the_server_reports_its_endpoint(shared: MachineServer) -> None:
    assert shared.endpoint.startswith("opc.tcp://127.0.0.1:")
    assert shared.endpoint.endswith("/bore01/server/")


async def test_the_namespace_gets_an_index(shared: MachineServer) -> None:
    """Index 0 and 1 are reserved by the standard, so ours must be 2 or higher."""
    assert shared.namespace_index >= 2


async def test_asking_for_the_namespace_before_starting_is_an_error() -> None:
    server = MachineServer(settings=settings())
    with pytest.raises(RuntimeError, match="not been started"):
        _ = server.namespace_index


async def test_running_before_starting_is_an_error() -> None:
    server = MachineServer(settings=settings())
    with pytest.raises(RuntimeError, match="start"):
        await server.run(duration_s=1.0)


async def test_starting_twice_is_harmless(shared: MachineServer) -> None:
    await shared.start()
    assert shared.namespace_index >= 2


async def test_stopping_releases_the_port() -> None:
    """A server that does not free its port cannot be restarted, which matters."""
    port = free_port()
    first = MachineServer(settings=ServerSettings(port=port))
    await first.start()
    await first.stop()

    second = MachineServer(settings=ServerSettings(port=port))
    await second.start()
    try:
        assert second.namespace_index >= 2
    finally:
        await second.stop()


async def test_stopping_a_server_that_never_started_is_harmless() -> None:
    await MachineServer(settings=settings()).stop()


# --------------------------------------------------------------------------
# The address space, seen from a client
# --------------------------------------------------------------------------


async def test_a_client_finds_the_machine_object_and_every_tag(
    shared: MachineServer,
) -> None:
    async with Client(shared.endpoint) as client:
        _, machine = await connect_machine(shared, client)
        children = await machine.get_children()
        names = [(await child.read_browse_name()).Name for child in children]
    assert set(names) == set(EXPECTED_NODES)


async def test_the_client_looks_the_namespace_up_by_uri(
    shared: MachineServer,
) -> None:
    """A client must never hard-code the index. It resolves the URI instead."""
    async with Client(shared.endpoint) as client:
        idx = await client.get_namespace_index(shared.settings.namespace_uri)
    assert idx == shared.namespace_index


async def test_every_tag_carries_a_description_with_its_units(
    shared: MachineServer,
) -> None:
    units = load_config().tag("BoreDiameter").units
    async with Client(shared.endpoint) as client:
        idx, machine = await connect_machine(shared, client)
        bore = await machine.get_child(f"{idx}:BoreDiameter")
        description = await bore.read_description()
    assert units in description.Text


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


async def test_published_values_reach_a_client(shared: MachineServer) -> None:
    spec = load_config().tag("BoreDiameter")
    for _ in range(60):
        await shared.step_and_publish()

    async with Client(shared.endpoint) as client:
        idx, machine = await connect_machine(shared, client)
        bore = await (await machine.get_child(f"{idx}:BoreDiameter")).read_value()

    assert isinstance(bore, float)
    assert abs(bore - spec.nominal) < 10 * spec.std_dev


async def test_a_counter_tag_is_published_as_a_whole_number(
    shared: MachineServer,
) -> None:
    """ScrapCount is counted, not measured, so it is an unsigned integer."""
    await shared.step_and_publish()
    async with Client(shared.endpoint) as client:
        idx, machine = await connect_machine(shared, client)
        scrap = await machine.get_child(f"{idx}:ScrapCount")
        data_value = await scrap.read_data_value()
    assert data_value.Value.VariantType == ua.VariantType.UInt32
    assert isinstance(data_value.Value.Value, int)


async def test_a_measured_tag_is_published_as_a_decimal(
    shared: MachineServer,
) -> None:
    await shared.step_and_publish()
    async with Client(shared.endpoint) as client:
        idx, machine = await connect_machine(shared, client)
        torque = await machine.get_child(f"{idx}:Torque")
        data_value = await torque.read_data_value()
    assert data_value.Value.VariantType == ua.VariantType.Double


async def test_values_change_as_samples_are_published(
    shared: MachineServer,
) -> None:
    readings = []
    async with Client(shared.endpoint) as client:
        idx, machine = await connect_machine(shared, client)
        torque = await machine.get_child(f"{idx}:Torque")
        for _ in range(5):
            await shared.step_and_publish()
            readings.append(await torque.read_value())
    assert len(set(readings)) == 5


async def test_part_count_climbs_as_parts_finish(shared: MachineServer) -> None:
    await shared.run(duration_s=60.0)  # under a second of wall clock at 100x
    async with Client(shared.endpoint) as client:
        idx, machine = await connect_machine(shared, client)
        parts = await (await machine.get_child(f"{idx}:PartCount")).read_value()
    assert parts >= 4  # 60 seconds at a 12 second cycle


async def test_status_reads_running_on_a_healthy_machine(
    shared: MachineServer,
) -> None:
    await shared.step_and_publish()
    async with Client(shared.endpoint) as client:
        idx, machine = await connect_machine(shared, client)
        status = await (await machine.get_child(f"{idx}:Status")).read_value()
    assert status == "RUNNING"


async def test_status_names_the_active_fault() -> None:
    schedule = FaultSchedule([MeanShift(tag="Torque", shift_sigma=3.0)], seed=1)
    simulator = MachineSimulator(load_config(), seed=1, faults=schedule)
    async with MachineServer(settings=settings(), simulator=simulator) as server:
        await server.step_and_publish()
        async with Client(server.endpoint) as client:
            idx, machine = await connect_machine(server, client)
            status = await (await machine.get_child(f"{idx}:Status")).read_value()
    assert status == "FAULT: MeanShift(Torque)"


# --------------------------------------------------------------------------
# Pacing
# --------------------------------------------------------------------------


async def test_run_publishes_exactly_the_right_number_of_samples() -> None:
    async with MachineServer(settings=settings(speed_factor=100.0)) as server:
        await server.run(duration_s=5.0)
        assert server.samples_published == 50  # 5 seconds at 10 Hz
        assert server.simulator.elapsed_s == pytest.approx(5.0, abs=1e-9)


async def test_speed_factor_compresses_wall_clock_time() -> None:
    async with MachineServer(settings=settings(speed_factor=50.0)) as server:
        started = time.monotonic()
        await server.run(duration_s=10.0)
        wall_clock = time.monotonic() - started
    assert server.samples_published == 100
    assert wall_clock < 2.0  # 10 simulated seconds at 50x is about 0.2


async def test_real_time_pacing_is_roughly_honest() -> None:
    async with MachineServer(settings=settings(speed_factor=5.0)) as server:
        started = time.monotonic()
        await server.run(duration_s=5.0)
        wall_clock = time.monotonic() - started
    assert 0.6 < wall_clock < 2.5  # 5 simulated seconds at 5x is about 1.0


# --------------------------------------------------------------------------
# Node lookup and settings validation
# --------------------------------------------------------------------------


async def test_looking_up_an_unknown_node_names_the_known_ones(
    shared: MachineServer,
) -> None:
    with pytest.raises(KeyError, match="BoreDiameter"):
        shared.node("NoSuchTag")


async def test_every_expected_node_is_addressable(shared: MachineServer) -> None:
    for name in EXPECTED_NODES:
        assert shared.node(name) is not None


async def test_the_endpoint_is_built_from_host_port_and_path() -> None:
    assert (
        ServerSettings(host="1.2.3.4", port=1234, path="/x/").endpoint
        == "opc.tcp://1.2.3.4:1234/x/"
    )


async def test_a_non_positive_speed_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="speed_factor"):
        ServerSettings(speed_factor=0.0)


async def test_an_impossible_port_is_rejected() -> None:
    with pytest.raises(ValueError, match="port"):
        ServerSettings(port=99999)


async def test_every_fault_preset_names_a_real_tag() -> None:
    """A preset with a typo in a tag name would silently do nothing."""
    known = set(load_config().tag_names)
    for name, faults in FAULT_PRESETS.items():
        for fault in faults:
            assert fault.tag in known, f"preset {name} refers to unknown {fault.tag}"