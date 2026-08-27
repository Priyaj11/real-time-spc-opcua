"""An OPC UA client that collects machine data by subscription.

There are two ways to get data out of an OPC UA server.

  Read (polling)
      The client asks "what is Torque right now?" and the server answers. To
      follow a 10 Hz tag you ask ten times a second, forever, whether or not
      anything changed. Simple, wasteful, and it misses anything that happens
      between two asks.

  Subscribe
      The client says once "tell me whenever Torque changes" and then waits.
      The server watches the value and pushes a notification. This is how real
      industrial data collection works, and it is what this module does.

One property of subscriptions surprises people and matters here: a server only
notifies on CHANGE. A sensor stuck at one value generates no traffic at all,
which looks identical to a dead connection. That is why the client tracks when
each tag was last heard from, not just its value.

This module knows nothing about control charts. It turns a network connection
into a stream of TagUpdate records and stops there.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Callable, Iterable, Sequence

from asyncua import Client, ua
from asyncua.common.node import Node

from spc_opcua.opcua_server import DEFAULT_PORT, NAMESPACE_URI

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = f"opc.tcp://127.0.0.1:{DEFAULT_PORT}/bore01/server/"
MACHINE_OBJECT = "Machine"

# Every variable the server publishes under the Machine object.
DEFAULT_TAGS: tuple[str, ...] = (
    "BoreDiameter",
    "Torque",
    "CycleTime",
    "Temperature",
    "Vibration",
    "ScrapCount",
    "PartCount",
    "Status",
)


@dataclass(frozen=True)
class TagUpdate:
    """One value arriving from the server.

    Attributes:
        tag: Which variable changed.
        value: The new value. Numbers for process tags, a string for Status.
        source_timestamp: When the server says the value was produced. This is
            the one to trust for ordering, not when we happened to receive it.
        received_at: Local monotonic clock reading when we got it, used for
            measuring how stale a tag has become.
    """

    tag: str
    value: float | int | str
    source_timestamp: datetime | None
    received_at: float

    @property
    def is_numeric(self) -> bool:
        """True for process tags, false for Status."""
        return isinstance(self.value, (int, float)) and not isinstance(
            self.value, bool
        )


class SubscriptionHandler:
    """Receives push notifications from the server.

    asyncua calls datachange_notification from inside its own receive loop, so
    this must return fast. It does one dictionary lookup and one queue put, and
    nothing else. Any real work belongs to whoever drains the queue.
    """

    def __init__(
        self,
        node_names: dict[Node, str],
        on_update: Callable[[TagUpdate], None],
    ) -> None:
        """Build a handler.

        Args:
            node_names: Maps each subscribed node to its tag name.
            on_update: Called once per notification.
        """
        self._node_names = node_names
        self._on_update = on_update
        self.dropped = 0

    def datachange_notification(
        self, node: Node, value: object, data: ua.DataChangeNotification
    ) -> None:
        """Called by asyncua whenever a subscribed value changes."""
        name = self._node_names.get(node)
        if name is None:
            self.dropped += 1
            return
        source_timestamp = None
        monitored = getattr(data, "monitored_item", None)
        if monitored is not None and monitored.Value is not None:
            source_timestamp = monitored.Value.SourceTimestamp
        self._on_update(
            TagUpdate(
                tag=name,
                value=value,
                source_timestamp=source_timestamp,
                received_at=time.monotonic(),
            )
        )

    def event_notification(self, event: object) -> None:  # pragma: no cover
        """Events are not used in this project, but asyncua may call this."""

    def status_change_notification(self, status: object) -> None:  # pragma: no cover
        """Server-side subscription status changes, logged and ignored."""
        logger.warning("Subscription status changed: %s", status)


@dataclass
class ClientSettings:
    """How the client connects and how eagerly it wants data.

    Attributes:
        endpoint: Address of the server.
        namespace_uri: URI to resolve into a namespace index. Never assume the
            index is 2; different servers number namespaces differently.
        publishing_interval_ms: How often the server is allowed to send a batch
            of notifications. Smaller means fresher data and more traffic. Half
            the sample period is a sensible default.
        queue_size: How many changes the server buffers per tag between
            batches. 1 means only the newest survives, which loses samples when
            the machine runs faster than the publishing interval.
        connect_timeout_s: How long to wait for the server to answer.
        max_queue: How many updates to buffer locally before dropping the
            oldest. A slow consumer must never make the network stall.
    """

    endpoint: str = DEFAULT_ENDPOINT
    namespace_uri: str = NAMESPACE_URI
    publishing_interval_ms: float = 50.0
    queue_size: int = 10
    connect_timeout_s: float = 10.0
    max_queue: int = 20_000


class MachineClient:
    """Connects to the machine server and collects tag values.

    Use it as an async context manager:

        async with MachineClient() as client:
            await client.subscribe()
            async for update in client.updates(limit=100):
                print(update.tag, update.value)
    """

    def __init__(
        self,
        settings: ClientSettings | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        """Build a client.

        Args:
            settings: Endpoint and subscription tuning.
            tags: Which variables to follow. All of them if omitted.
        """
        self.settings = settings if settings is not None else ClientSettings()
        self.tags: tuple[str, ...] = tuple(tags) if tags else DEFAULT_TAGS

        self._client: Client | None = None
        self._namespace_index: int | None = None
        self._machine: Node | None = None
        self._nodes: dict[str, Node] = {}
        self._subscription = None
        self._handler: SubscriptionHandler | None = None
        self._queue: asyncio.Queue[TagUpdate] = asyncio.Queue()
        self._latest: dict[str, TagUpdate] = {}
        self._update_count = 0
        self._dropped = 0
        self._connected = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True between connect() and disconnect()."""
        return self._connected

    @property
    def namespace_index(self) -> int:
        """Index the server assigned to our namespace URI."""
        if self._namespace_index is None:
            raise RuntimeError("Client is not connected")
        return self._namespace_index

    @property
    def update_count(self) -> int:
        """How many notifications have arrived since connecting."""
        return self._update_count

    @property
    def dropped_count(self) -> int:
        """Updates thrown away because the local buffer was full."""
        return self._dropped

    @property
    def latest(self) -> dict[str, float | int | str]:
        """The most recent value seen for every tag heard from so far."""
        return {name: update.value for name, update in self._latest.items()}

    def age_s(self, tag: str) -> float | None:
        """Seconds since this tag last changed, or None if never heard from.

        A large age is not automatically a problem. Status changes rarely by
        design. But a process tag that has not moved in ten seconds means
        either a stuck sensor or a broken connection, and the dashboard needs
        to be able to say so.
        """
        update = self._latest.get(tag)
        if update is None:
            return None
        return time.monotonic() - update.received_at

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection and resolve every tag node."""
        if self._connected:
            return

        client = Client(
            url=self.settings.endpoint, timeout=self.settings.connect_timeout_s
        )
        await client.connect()
        try:
            idx = await client.get_namespace_index(self.settings.namespace_uri)
            machine = await client.nodes.objects.get_child(f"{idx}:{MACHINE_OBJECT}")
            nodes = {}
            for name in self.tags:
                nodes[name] = await machine.get_child(f"{idx}:{name}")
        except Exception:
            await client.disconnect()
            raise

        self._client = client
        self._namespace_index = idx
        self._machine = machine
        self._nodes = nodes
        self._connected = True

        logger.info(
            "Connected to %s, namespace %d, following %d tags",
            self.settings.endpoint,
            idx,
            len(nodes),
        )

    async def disconnect(self) -> None:
        """Cancel the subscription and close the connection."""
        if not self._connected or self._client is None:
            return
        if self._subscription is not None:
            try:
                await self._subscription.delete()
            except Exception as exc:  # pragma: no cover
                logger.debug("Subscription already gone: %s", exc)
            self._subscription = None
        await self._client.disconnect()
        self._connected = False
        logger.info(
            "Disconnected after %d updates (%d dropped)",
            self._update_count,
            self._dropped,
        )

    async def __aenter__(self) -> "MachineClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Reading, the polling way
    # ------------------------------------------------------------------

    async def read(self, tag: str) -> float | int | str:
        """Ask the server for one value right now.

        Raises:
            KeyError: if the tag is not being followed.
        """
        self._require_connected()
        try:
            node = self._nodes[tag]
        except KeyError as exc:
            raise KeyError(
                f"Not following {tag!r}. Following: {tuple(self._nodes)}"
            ) from exc
        return await node.read_value()

    async def read_all(self) -> dict[str, float | int | str]:
        """Ask the server for every followed value right now.

        This is the polling approach, kept for comparison and for a one-off
        snapshot at startup. The continuous path is subscribe().
        """
        self._require_connected()
        return {name: await node.read_value() for name, node in self._nodes.items()}

    # ------------------------------------------------------------------
    # Subscribing, the push way
    # ------------------------------------------------------------------

    async def subscribe(self, tags: Iterable[str] | None = None) -> list[int]:
        """Ask the server to push changes for the given tags.

        Args:
            tags: Which tags to follow. Every followed tag if omitted.

        Returns:
            The monitored item handles the server assigned, one per tag.
        """
        self._require_connected()
        assert self._client is not None

        wanted = tuple(tags) if tags is not None else self.tags
        nodes = [self._nodes[name] for name in wanted]
        node_names = {self._nodes[name]: name for name in wanted}

        self._handler = SubscriptionHandler(node_names, self._enqueue)
        self._subscription = await self._client.create_subscription(
            self.settings.publishing_interval_ms, self._handler
        )
        handles = await self._subscription.subscribe_data_change(
            nodes, queuesize=self.settings.queue_size
        )
        logger.info(
            "Subscribed to %d tags, publishing interval %.0f ms, queue size %d",
            len(nodes),
            self.settings.publishing_interval_ms,
            self.settings.queue_size,
        )
        return list(handles) if isinstance(handles, list) else [handles]

    def _enqueue(self, update: TagUpdate) -> None:
        """Record an update. Called from asyncua's receive loop, so keep it cheap."""
        self._update_count += 1
        self._latest[update.tag] = update
        if self._queue.qsize() >= self.settings.max_queue:
            # Drop the oldest rather than block. A slow consumer must never
            # make the server's connection stall.
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
        self._queue.put_nowait(update)

    async def updates(
        self, limit: int | None = None, timeout_s: float | None = None
    ) -> AsyncIterator[TagUpdate]:
        """Yield updates as they arrive.

        Args:
            limit: Stop after this many updates, or None for no limit.
            timeout_s: Give up if nothing arrives for this long. None waits
                forever, which is what a live dashboard wants.
        """
        yielded = 0
        while limit is None or yielded < limit:
            if timeout_s is None:
                update = await self._queue.get()
            else:
                try:
                    update = await asyncio.wait_for(self._queue.get(), timeout_s)
                except asyncio.TimeoutError:
                    return
            yield update
            yielded += 1

    async def collect(
        self, duration_s: float, limit: int | None = None
    ) -> list[TagUpdate]:
        """Gather every update that arrives over a window of wall-clock time."""
        collected: list[TagUpdate] = []
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if limit is not None and len(collected) >= limit:
                break
            remaining = deadline - time.monotonic()
            try:
                collected.append(
                    await asyncio.wait_for(self._queue.get(), max(remaining, 0.001))
                )
            except asyncio.TimeoutError:
                break
        return collected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Call connect() before using the client")


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Command line options for the client demonstration."""
    parser = argparse.ArgumentParser(
        prog="python -m spc_opcua.opcua_client",
        description=(
            "Collect BORE-01 data over OPC UA. With no --endpoint, starts its "
            "own server so the demonstration is self-contained."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Server to connect to. Omit to start one internally.",
    )
    parser.add_argument(
        "--seconds", type=float, default=6.0, help="Wall-clock seconds to collect"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="Speed factor for the internal server (ignored with --endpoint)",
    )
    parser.add_argument(
        "--fault", default="tool-wear", help="Fault preset for the internal server"
    )
    return parser


async def demonstrate(client: MachineClient, seconds: float) -> None:
    """Show a polled snapshot, then subscribe and report what arrived."""
    snapshot = await client.read_all()
    print("\nOne-shot READ of every tag (the polling way)")
    print("-" * 52)
    for name, value in snapshot.items():
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {name:<14} {shown}")

    await client.subscribe()
    print(f"\nSUBSCRIBED. Collecting for {seconds:.0f} s, server pushing to us.\n")
    updates = await client.collect(seconds)

    per_tag: dict[str, int] = {}
    for update in updates:
        per_tag[update.tag] = per_tag.get(update.tag, 0) + 1

    print(f"{'TAG':<14}{'UPDATES':>9}{'PER SECOND':>13}{'LAST VALUE':>16}")
    print("-" * 52)
    for name in DEFAULT_TAGS:
        count = per_tag.get(name, 0)
        value = client.latest.get(name, "-")
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"{name:<14}{count:>9}{count / seconds:>13.1f}{shown:>16}")

    print(f"\nTotal notifications: {len(updates)} in {seconds:.0f} s")
    print(
        "Tags that barely notify are not broken. A subscription only fires on "
        "CHANGE,\nand ScrapCount, PartCount and Status rarely change."
    )


async def run_demo(args: argparse.Namespace) -> None:
    """Connect to a server, or start one, and run the demonstration."""
    if args.endpoint:
        async with MachineClient(ClientSettings(endpoint=args.endpoint)) as client:
            await demonstrate(client, args.seconds)
        return

    # Self-contained mode: run our own server in the same event loop.
    import socket

    from spc_opcua.config import load_config
    from spc_opcua.opcua_server import FAULT_PRESETS, MachineServer, ServerSettings
    from spc_opcua.simulator.faults import FaultSchedule
    from spc_opcua.simulator.machine import MachineSimulator

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    config = load_config()
    schedule = FaultSchedule(FAULT_PRESETS[args.fault], seed=config.random_seed)
    simulator = MachineSimulator(config, faults=schedule)
    settings = ServerSettings(port=port, speed_factor=args.speed)

    async with MachineServer(config, settings, simulator) as server:
        publisher = asyncio.create_task(server.run())
        try:
            async with MachineClient(
                ClientSettings(endpoint=server.endpoint)
            ) as client:
                print(
                    f"\nInternal server on {server.endpoint} "
                    f"(scenario: {args.fault})"
                )
                await demonstrate(client, args.seconds)
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass


def main() -> None:
    """Entry point for the client demonstration."""
    from spc_opcua.logging_setup import configure_logging

    configure_logging()
    args = build_parser().parse_args()
    try:
        asyncio.run(run_demo(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()