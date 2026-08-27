"""Turns a stream of OPC UA tag updates into parts and subgroups.

The client hands over one TagUpdate at a time, tag by tag, in whatever order
the server batched them. The SPC engine wants subgroups of five consecutive
parts. This module is the piece in between.

How a part is recognised
------------------------
A subscription only fires on CHANGE, and the machine holds BoreDiameter steady
between parts. So one BoreDiameter notification is exactly one finished part.
That is the whole rule, and it falls straight out of the two-clock model from
Milestone 2.

Because it would be quietly disastrous to be wrong about that, the collector
also watches PartCount and compares. If the machine says it has made 300 parts
and we have recorded 297 measurements, three went missing and the collector
says so rather than silently reporting statistics on incomplete data.

Nothing here computes a control limit. It produces parts, subgroups and a
table, and stops.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from spc_opcua.config import MachineConfig, load_config
from spc_opcua.opcua_client import MachineClient, TagUpdate
from spc_opcua.spc.subgroups import Subgroup, SubgroupBuilder

logger = logging.getLogger(__name__)

DEFAULT_CHART_TAG = "BoreDiameter"

# Tags whose latest value is snapshotted onto every part record.
CONTEXT_TAGS: tuple[str, ...] = (
    "Torque",
    "CycleTime",
    "Temperature",
    "Vibration",
    "ScrapCount",
    "Status",
)


@dataclass(frozen=True)
class PartRecord:
    """One finished part, with the machine's state at the moment it finished.

    Attributes:
        sequence: Our own count of measurements received, from zero.
        part_index: The machine's own PartCount when this part was recorded.
        value: The charted measurement, normally BoreDiameter.
        timestamp: The server's source timestamp for that measurement.
        context: Latest value of every other tag when the part finished.
    """

    sequence: int
    part_index: int
    value: float
    timestamp: datetime | None
    context: dict[str, float | int | str] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        """Flatten into one dictionary, suitable for a table or a CSV row."""
        row: dict[str, object] = {
            "sequence": self.sequence,
            "part_index": self.part_index,
            "timestamp": self.timestamp,
            "value": self.value,
        }
        row.update(self.context)
        return row


class DataCollector:
    """Assembles tag updates into part records and subgroups.

    Example:
        >>> from spc_opcua.config import load_config
        >>> collector = DataCollector(load_config())
        >>> collector.subgroup_size
        5
    """

    def __init__(
        self,
        config: MachineConfig | None = None,
        chart_tag: str = DEFAULT_CHART_TAG,
        subgroup_size: int | None = None,
        warmup_parts: int = 0,
        skip_initial_snapshot: bool = True,
    ) -> None:
        """Build a collector.

        Args:
            config: The machine definition. Loaded from machine.yaml if omitted.
            chart_tag: Which measurement drives part detection and the charts.
            subgroup_size: Parts per subgroup. Taken from the config if omitted.
            warmup_parts: Skip this many parts at the start before collecting.
                A spindle warming up is not a stable process, and control
                limits built from unstable data are meaningless. Milestone 7
                revisits this.
            skip_initial_snapshot: Ignore the very first reading of the charted
                tag. When a client subscribes, the server immediately sends the
                CURRENT value of every monitored item. That first notification
                is a snapshot of a part already finished, not a new one, and it
                arrives before the other tags so it carries no context. Set
                False when feeding the collector from something other than a
                fresh subscription.
        """
        self.config = config if config is not None else load_config()
        self.chart_tag = chart_tag
        self.subgroup_size = (
            subgroup_size if subgroup_size is not None else self.config.subgroup_size
        )
        self.warmup_parts = max(0, warmup_parts)
        self.skip_initial_snapshot = skip_initial_snapshot

        # Fail early if the tag does not exist, rather than collecting nothing.
        self.spec = self.config.tag(chart_tag)

        self._builder = SubgroupBuilder(tag=chart_tag, size=self.subgroup_size)
        self._parts: list[PartRecord] = []
        self._subgroups: list[Subgroup] = []
        self._latest: dict[str, float | int | str] = {}
        self._received = 0
        self._skipped_warmup = 0
        self._skipped_snapshot = False
        self._last_part_count: int | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parts(self) -> tuple[PartRecord, ...]:
        """Every part recorded so far."""
        return tuple(self._parts)

    @property
    def subgroups(self) -> tuple[Subgroup, ...]:
        """Every complete subgroup so far."""
        return tuple(self._subgroups)

    @property
    def latest(self) -> dict[str, float | int | str]:
        """Most recent value seen for every tag."""
        return dict(self._latest)

    @property
    def updates_received(self) -> int:
        """How many TagUpdates have been handed to this collector."""
        return self._received

    @property
    def initial_snapshot_skipped(self) -> bool:
        """True once the subscription's opening snapshot has been discarded."""
        return self._skipped_snapshot

    @property
    def parts_skipped_during_warmup(self) -> int:
        """Parts deliberately discarded before collection began."""
        return self._skipped_warmup

    @property
    def pending_parts(self) -> int:
        """Parts waiting for the current subgroup to fill."""
        return len(self._builder)

    @property
    def machine_part_count(self) -> int | None:
        """The machine's own PartCount, as last reported."""
        return self._last_part_count

    @property
    def missed_parts(self) -> int:
        """Parts the machine says it made that we never received a value for.

        Anything above zero means the collector is reporting statistics on
        incomplete data, which is worth an alarm rather than a shrug.
        """
        if self._last_part_count is None:
            return 0
        expected = self._last_part_count - self._skipped_warmup
        return max(0, expected - len(self._parts))

    # ------------------------------------------------------------------
    # Collecting
    # ------------------------------------------------------------------

    def handle(self, update: TagUpdate) -> Subgroup | None:
        """Take one tag update.

        Args:
            update: A value that arrived from the server.

        Returns:
            A completed Subgroup if this update filled one, else None.
        """
        self._received += 1
        self._latest[update.tag] = update.value

        if update.tag == "PartCount":
            self._last_part_count = int(update.value)
            return None

        if update.tag != self.chart_tag:
            return None

        if self.skip_initial_snapshot and not self._skipped_snapshot:
            # The opening notification of a subscription is the current value,
            # not a change. Recording it would invent a part.
            self._skipped_snapshot = True
            logger.debug(
                "Discarded the subscription's opening %s snapshot", self.chart_tag
            )
            return None

        if self._skipped_warmup < self.warmup_parts:
            self._skipped_warmup += 1
            return None

        record = PartRecord(
            sequence=len(self._parts),
            part_index=int(self._latest.get("PartCount", -1)),
            value=float(update.value),
            timestamp=update.source_timestamp,
            context={
                name: self._latest[name]
                for name in CONTEXT_TAGS
                if name in self._latest
            },
        )
        self._parts.append(record)

        subgroup = self._builder.add(
            record.value, part_index=record.part_index, timestamp=record.timestamp
        )
        if subgroup is not None:
            self._subgroups.append(subgroup)
            logger.debug(
                "Subgroup %d complete: mean %.5f, range %.5f",
                subgroup.index,
                subgroup.mean,
                subgroup.range,
            )
        return subgroup

    def handle_many(self, updates: Iterable[TagUpdate]) -> list[Subgroup]:
        """Take many updates and return every subgroup they completed."""
        completed = []
        for update in updates:
            subgroup = self.handle(update)
            if subgroup is not None:
                completed.append(subgroup)
        return completed

    def mark_stoppage(self) -> None:
        """Tell the collector that production was interrupted.

        Parts from either side of a break are not consecutive under the same
        conditions, so the partial subgroup is discarded rather than bridged.
        """
        self._builder.discard_partial()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def parts_frame(self) -> pd.DataFrame:
        """Every part as a pandas table, one row per part."""
        if not self._parts:
            return pd.DataFrame(
                columns=["sequence", "part_index", "timestamp", "value"]
            )
        return pd.DataFrame([record.as_row() for record in self._parts])

    def subgroups_frame(self) -> pd.DataFrame:
        """Every subgroup as a pandas table, one row per subgroup."""
        if not self._subgroups:
            return pd.DataFrame(
                columns=["subgroup", "tag", "n", "mean", "range", "min", "max"]
            )
        return pd.DataFrame([group.as_row() for group in self._subgroups])

    def write_parts_csv(self, path: Path | str) -> Path:
        """Save the part table to a CSV file, creating the folder if needed."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.parts_frame().to_csv(destination, index=False)
        logger.info("Wrote %d parts to %s", len(self._parts), destination)
        return destination

    def write_subgroups_csv(self, path: Path | str) -> Path:
        """Save the subgroup table to a CSV file, creating the folder if needed."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.subgroups_frame().to_csv(destination, index=False)
        logger.info("Wrote %d subgroups to %s", len(self._subgroups), destination)
        return destination

    def summary(self) -> str:
        """A short human readable report of what was collected."""
        frame = self.subgroups_frame()
        lines = [
            f"Tag            : {self.chart_tag}",
            f"Updates seen   : {self._received}",
            f"Parts recorded : {len(self._parts)}",
            f"Machine count  : {self._last_part_count}",
            f"Missed parts   : {self.missed_parts}",
            f"Subgroups (n={self.subgroup_size}): {len(self._subgroups)}"
            f", {self.pending_parts} part(s) pending",
        ]
        if not frame.empty:
            lines += [
                f"Mean of means  : {frame['mean'].mean():.5f} {self.spec.units}",
                f"Mean range     : {frame['range'].mean():.5f} {self.spec.units}",
            ]
        return "\n".join(lines)


async def collect_from_client(
    client: MachineClient,
    collector: DataCollector,
    duration_s: float | None = None,
    subgroup_limit: int | None = None,
) -> list[Subgroup]:
    """Drain a subscribed client into a collector.

    Args:
        client: An already connected and subscribed client.
        collector: Where the updates go.
        duration_s: Wall-clock seconds to keep collecting, or None for no limit.
        subgroup_limit: Stop once this many subgroups are complete.

    Returns:
        Every subgroup completed during this call.
    """
    completed: list[Subgroup] = []
    loop = asyncio.get_running_loop()
    deadline = None if duration_s is None else loop.time() + duration_s

    while True:
        if subgroup_limit is not None and len(completed) >= subgroup_limit:
            break
        if deadline is None:
            timeout = None
        else:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break

        found = False
        async for update in client.updates(limit=1, timeout_s=timeout):
            found = True
            subgroup = collector.handle(update)
            if subgroup is not None:
                completed.append(subgroup)
        if not found:
            break

    return completed


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------


def build_parser() -> "argparse.ArgumentParser":
    """Command line options for the pipeline demonstration."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m spc_opcua.pipeline",
        description=(
            "Collect BORE-01 data over OPC UA and assemble it into subgroups. "
            "Starts its own server, so nothing else needs to be running."
        ),
    )
    parser.add_argument(
        "--subgroups", type=int, default=24, help="How many subgroups to collect"
    )
    parser.add_argument(
        "--speed", type=float, default=120.0, help="Server speed factor"
    )
    parser.add_argument(
        "--fault", default="tool-wear", help="Fault preset for the server"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Parts to discard before collecting (spindle warm-up)",
    )
    parser.add_argument(
        "--out",
        default="data",
        help="Folder to write sample_data.csv and subgroups.csv into",
    )
    return parser


async def run_pipeline(args) -> None:
    """Start a server, subscribe, collect subgroups, and save the tables."""
    import socket

    from spc_opcua.opcua_client import ClientSettings
    from spc_opcua.opcua_server import FAULT_PRESETS, MachineServer, ServerSettings
    from spc_opcua.simulator.faults import FaultSchedule
    from spc_opcua.simulator.machine import MachineSimulator

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    config = load_config()
    schedule = FaultSchedule(FAULT_PRESETS[args.fault], seed=config.random_seed)
    simulator = MachineSimulator(config, faults=schedule)
    server = MachineServer(
        config, ServerSettings(port=port, speed_factor=args.speed), simulator
    )
    collector = DataCollector(config, warmup_parts=args.warmup)

    async with server:
        publisher = asyncio.create_task(server.run())
        try:
            async with MachineClient(
                ClientSettings(endpoint=server.endpoint)
            ) as client:
                await client.subscribe()
                print(
                    f"\nCollecting {args.subgroups} subgroups of "
                    f"{collector.subgroup_size} parts, scenario {args.fault}, "
                    f"speed x{args.speed:g}\n"
                )
                await collect_from_client(
                    client,
                    collector,
                    duration_s=180.0,
                    subgroup_limit=args.subgroups,
                )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass

    print(collector.summary())

    frame = collector.subgroups_frame()
    if not frame.empty:
        shown = frame[["subgroup", "n", "mean", "range", "min", "max"]].round(5)
        print("\n" + shown.to_string(index=False))

    out = Path(args.out)
    collector.write_parts_csv(out / "sample_data.csv")
    collector.write_subgroups_csv(out / "subgroups.csv")
    print(f"\nSaved {out / 'sample_data.csv'} and {out / 'subgroups.csv'}\n")


def main() -> None:
    """Entry point for the pipeline demonstration."""
    from spc_opcua.logging_setup import configure_logging

    configure_logging()
    args = build_parser().parse_args()
    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()