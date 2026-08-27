"""An OPC UA server that publishes the simulated machine onto the network.

This is where the project stops being a Python script and starts being an
industrial system. Everything up to now ran as fast as the processor allowed.
From here there is a clock on the wall, a network socket, and a client that
could be written in any language on any machine.

The address space looks like this:

    Objects
    └── Machine                (BrowseName BORE-01's Machine folder)
        ├── BoreDiameter       Double, mm
        ├── Torque             Double, Nm
        ├── CycleTime          Double, s
        ├── Temperature        Double, degC
        ├── Vibration          Double, mm/s
        ├── ScrapCount         UInt32, parts
        ├── PartCount          UInt32, parts
        └── Status             String

Nothing in here knows what a control chart is. It takes Samples from the
simulator and writes them to nodes. That is the whole job.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass

from asyncua import Server, ua
from asyncua.common.node import Node

from spc_opcua.config import MachineConfig, load_config
from spc_opcua.simulator.faults import (
    Fault,
    FaultSchedule,
    MeanShift,
    Outlier,
    SensorDrift,
    SensorStuck,
    ToolWear,
    VarianceInflation,
)
from spc_opcua.simulator.machine import MachineSimulator, Sample

logger = logging.getLogger(__name__)

DEFAULT_PORT = 4840
NAMESPACE_URI = "http://bore01.local/spc-opcua"

# Tags published as whole numbers rather than decimals.
COUNTER_TAGS = frozenset({"ScrapCount"})


@dataclass(frozen=True)
class ServerSettings:
    """How the server presents itself and how fast it runs.

    Attributes:
        host: Address to listen on. 127.0.0.1 keeps it on this machine only.
        port: TCP port. 4840 is the registered OPC UA port.
        path: Endpoint path, appended after host and port.
        server_name: Name a browsing client sees for this server.
        namespace_uri: Unique identifier for our own namespace, so our node
            names cannot collide with another vendor's.
        speed_factor: 1.0 runs in real time. 10.0 simulates ten seconds of
            production per wall-clock second, which is how you watch an hour
            of tool wear over a coffee.
        machine_object_name: BrowseName of the folder holding the tags.
    """

    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    path: str = "/bore01/server/"
    server_name: str = "BORE-01 SPC Demo Server"
    namespace_uri: str = NAMESPACE_URI
    speed_factor: float = 1.0
    machine_object_name: str = "Machine"

    def __post_init__(self) -> None:
        if self.speed_factor <= 0.0:
            raise ValueError("speed_factor must be greater than zero")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

    @property
    def endpoint(self) -> str:
        """Full address a client connects to."""
        return f"opc.tcp://{self.host}:{self.port}{self.path}"


class MachineServer:
    """Publishes a MachineSimulator's output as OPC UA variables.

    Use it as an async context manager, which guarantees the socket is closed
    even if the publishing loop raises:

        async with MachineServer() as server:
            await server.run(duration_s=30.0)
    """

    def __init__(
        self,
        config: MachineConfig | None = None,
        settings: ServerSettings | None = None,
        simulator: MachineSimulator | None = None,
    ) -> None:
        """Build a server.

        Args:
            config: The machine definition. Loaded from machine.yaml if omitted.
            settings: Endpoint and pacing. Sensible defaults if omitted.
            simulator: Where the data comes from. A healthy machine if omitted.
        """
        self.config = config if config is not None else load_config()
        self.settings = settings if settings is not None else ServerSettings()
        self.simulator = (
            simulator
            if simulator is not None
            else MachineSimulator(self.config)
        )

        self._server: Server | None = None
        self._namespace_index: int | None = None
        self._machine_node: Node | None = None
        self._nodes: dict[str, Node] = {}
        self._started = False
        self._samples_published = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """Address a client should connect to."""
        return self.settings.endpoint

    @property
    def namespace_index(self) -> int:
        """Numeric index our namespace was given by the server.

        OPC UA identifies a namespace by a text URI, but node identifiers use a
        short integer index instead, assigned when the namespace is registered.
        A client looks the index up by URI rather than assuming it, because the
        number can differ between servers.
        """
        if self._namespace_index is None:
            raise RuntimeError("Server has not been started yet")
        return self._namespace_index

    @property
    def namespace_uri_and_index(self) -> str:
        """Our namespace URI together with the index this server gave it."""
        return f"{self.settings.namespace_uri} (index {self.namespace_index})"

    @property
    def samples_published(self) -> int:
        """How many samples have been written to the address space."""
        return self._samples_published

    @property
    def tag_node_names(self) -> tuple[str, ...]:
        """Every variable name under the Machine object."""
        return tuple(self._nodes.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build the address space and open the network socket."""
        if self._started:
            return

        server = Server()
        await server.init()
        server.set_endpoint(self.settings.endpoint)
        server.set_server_name(self.settings.server_name)
        # No encryption: this is a local demo. A production server would load a
        # certificate and private key here and restrict the security policies.
        server.set_security_policy([ua.SecurityPolicyType.NoSecurity])

        idx = await server.register_namespace(self.settings.namespace_uri)
        machine = await server.nodes.objects.add_object(
            idx, self.settings.machine_object_name
        )
        await self._add_tag_variables(idx, machine)

        await server.start()

        self._server = server
        self._namespace_index = idx
        self._machine_node = machine
        self._started = True

        logger.info(
            "OPC UA server listening on %s, namespace %d, %d variables",
            self.settings.endpoint,
            idx,
            len(self._nodes),
        )

    async def _add_tag_variables(self, idx: int, machine: Node) -> None:
        """Create one variable node per tag, plus PartCount and Status."""
        for spec in self.config.tags:
            if spec.name in COUNTER_TAGS:
                initial = ua.Variant(0, ua.VariantType.UInt32)
            else:
                initial = ua.Variant(float(spec.nominal), ua.VariantType.Double)
            node = await machine.add_variable(idx, spec.name, initial)
            await node.write_attribute(
                ua.AttributeIds.Description,
                ua.DataValue(
                    ua.Variant(
                        ua.LocalizedText(f"{spec.description} [{spec.units}]"),
                        ua.VariantType.LocalizedText,
                    )
                ),
            )
            self._nodes[spec.name] = node

        self._nodes["PartCount"] = await machine.add_variable(
            idx, "PartCount", ua.Variant(0, ua.VariantType.UInt32)
        )
        self._nodes["Status"] = await machine.add_variable(
            idx, "Status", ua.Variant("STOPPED", ua.VariantType.String)
        )

    async def stop(self) -> None:
        """Close the socket and release the port."""
        if not self._started or self._server is None:
            return
        await self._write_status("STOPPED")
        await self._server.stop()
        self._started = False
        logger.info(
            "OPC UA server stopped after publishing %d samples",
            self._samples_published,
        )

    async def __aenter__(self) -> "MachineServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def node(self, name: str) -> Node:
        """Look up one variable node by name.

        Raises:
            KeyError: if no such variable exists.
        """
        try:
            return self._nodes[name]
        except KeyError as exc:
            raise KeyError(
                f"No node named {name!r}. Known: {self.tag_node_names}"
            ) from exc

    async def _write_status(self, status: str) -> None:
        """Update the Status string, ignoring a server that is already down."""
        node = self._nodes.get("Status")
        if node is None:
            return
        await node.write_value(ua.Variant(status, ua.VariantType.String))

    async def publish(self, sample: Sample) -> None:
        """Write one sample's published values into the address space."""
        for name, value in sample.values.items():
            node = self._nodes.get(name)
            if node is None:
                continue
            if name in COUNTER_TAGS:
                await node.write_value(
                    ua.Variant(int(round(value)), ua.VariantType.UInt32)
                )
            else:
                await node.write_value(
                    ua.Variant(float(value), ua.VariantType.Double)
                )

        await self._nodes["PartCount"].write_value(
            ua.Variant(int(sample.part_index), ua.VariantType.UInt32)
        )
        status = (
            "FAULT: " + ", ".join(sample.active_faults)
            if sample.is_faulted
            else "RUNNING"
        )
        await self._write_status(status)
        self._samples_published += 1

    async def step_and_publish(self) -> Sample:
        """Advance the simulator one sample and publish it."""
        sample = self.simulator.step()
        await self.publish(sample)
        return sample

    async def run(self, duration_s: float | None = None) -> None:
        """Publish at the configured rate until the duration elapses.

        Args:
            duration_s: Simulated seconds to run for, or None to run forever
                until cancelled.

        Note:
            The deadline is advanced by a fixed period rather than sleeping a
            fixed amount each time. Sleeping a fixed amount accumulates the cost
            of the work itself, so a 10 Hz loop slowly becomes 9.7 Hz. Chasing
            an absolute deadline keeps the long-run rate exact.
        """
        if not self._started:
            raise RuntimeError("Call start() before run()")

        period = self.config.sample_period_s / self.settings.speed_factor
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        remaining = (
            None
            if duration_s is None
            else int(round(duration_s * self.config.sample_rate_hz))
        )

        logger.info(
            "Publishing at %.1f Hz x%.4g speed (%.1f ms per sample)",
            self.config.sample_rate_hz,
            self.settings.speed_factor,
            period * 1000.0,
        )

        while remaining is None or remaining > 0:
            await self.step_and_publish()
            if remaining is not None:
                remaining -= 1
            deadline += period
            delay = deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # Fell behind. Reset the deadline rather than sprinting to
                # catch up, which would publish a burst of samples at once.
                deadline = loop.time()


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------


FAULT_PRESETS: dict[str, list[Fault]] = {
    "none": [],
    "tool-wear": [
        ToolWear(tag="BoreDiameter", start_s=120.0, rate_per_hour=-0.045),
        ToolWear(tag="Torque", start_s=120.0, rate_per_hour=3.0),
    ],
    "mean-shift": [MeanShift(tag="BoreDiameter", start_s=300.0, shift_sigma=2.5)],
    "outliers": [
        Outlier(
            tag="BoreDiameter", start_s=120.0, probability=0.08, magnitude_sigma=6.0
        )
    ],
    "variance": [VarianceInflation(tag="BoreDiameter", start_s=300.0, factor=2.5)],
    "sensor-drift": [
        SensorDrift(tag="BoreDiameter", start_s=120.0, rate_per_hour=0.05)
    ],
    "sensor-stuck": [SensorStuck(tag="Torque", start_s=180.0)],
}


def build_parser() -> argparse.ArgumentParser:
    """Command line options for running the server."""
    parser = argparse.ArgumentParser(
        prog="python -m spc_opcua.opcua_server",
        description="Publish the simulated BORE-01 machining station over OPC UA.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="TCP port (default 4840)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Simulated seconds per wall-clock second (default 1.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulated seconds to run for (default: forever)",
    )
    parser.add_argument(
        "--fault",
        choices=sorted(FAULT_PRESETS),
        default="none",
        help="Fault scenario to inject (default none)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    return parser


async def serve(args: argparse.Namespace) -> None:
    """Build everything from parsed arguments and run until stopped."""
    config = load_config()
    schedule = FaultSchedule(
        FAULT_PRESETS[args.fault], seed=args.seed or config.random_seed
    )
    simulator = MachineSimulator(config, seed=args.seed, faults=schedule)
    settings = ServerSettings(port=args.port, speed_factor=args.speed)

    async with MachineServer(config, settings, simulator) as server:
        print(f"\nEndpoint  : {server.endpoint}")
        print(f"Namespace : {server.namespace_uri_and_index}")
        print(f"Variables : {', '.join(server.tag_node_names)}")
        print(f"Scenario  : {args.fault}")
        print("\nPress Control+C to stop.\n")
        await server.run(duration_s=args.duration)
        print(
            f"Finished. {server.samples_published} samples, "
            f"{simulator.parts_completed} parts, {simulator.scrap_count} scrapped."
        )


def main() -> None:
    """Entry point. Run the server until the duration elapses or Control+C."""
    from spc_opcua.logging_setup import configure_logging

    configure_logging()
    args = build_parser().parse_args()
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()