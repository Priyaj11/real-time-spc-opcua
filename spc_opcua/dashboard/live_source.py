"""A live data source that Streamlit can safely read from.

Streamlit's execution model is the problem this module solves. Every time the
page refreshes or a widget changes, Streamlit re-runs your script from the
first line to the last. Nothing survives between runs except what you put in
st.session_state.

An OPC UA connection cannot live like that. It is a long-running asyncio
conversation with a server, and restarting it on every page refresh would
reconnect several times a second.

So the connection lives in its own thread, with its own event loop, running
continuously. It owns the client, the data collector and the SPC engine, and it
keeps a plain snapshot of the current state behind a lock. Streamlit re-runs as
often as it likes and simply reads that snapshot.

Two rules make this safe:

    Only the background thread ever writes.
    Streamlit only ever reads, and reads a frozen copy.

Nothing here imports Streamlit, so it can be tested headlessly.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from spc_opcua.config import MachineConfig, load_config
from spc_opcua.opcua_client import ClientSettings, MachineClient
from spc_opcua.opcua_server import FAULT_PRESETS, MachineServer, ServerSettings
from spc_opcua.pipeline import DataCollector
from spc_opcua.simulator.faults import FaultSchedule
from spc_opcua.simulator.machine import MachineSimulator
from spc_opcua.spc.alarms import Alarm
from spc_opcua.spc.capability import Capability
from spc_opcua.spc.control_charts import ChartLimits
from spc_opcua.spc.engine import SPCEngine

logger = logging.getLogger(__name__)

# How often the background thread refreshes the live tag values in the
# snapshot. Chart data is refreshed whenever a subgroup completes, which is far
# less often.
SNAPSHOT_INTERVAL_S = 0.25

# How many individual part measurements the snapshot carries, for the chart
# that plots parts against the specification limits.
RECENT_PARTS = 200


@dataclass(frozen=True)
class Snapshot:
    """A frozen view of the whole system, safe to hand to the user interface.

    Everything here is a plain value or an immutable object. Nothing the
    background thread will mutate afterwards.
    """

    connected: bool = False
    error: str | None = None
    endpoint: str = ""
    scenario: str = "none"
    speed: float = 1.0

    machine_status: str = "STOPPED"
    latest: dict[str, Any] = field(default_factory=dict)
    ages: dict[str, float | None] = field(default_factory=dict)

    parts: int = 0
    recent_parts: tuple[float, ...] = ()
    scrap: int = 0
    missed: int = 0
    updates: int = 0

    phase: str = "baseline"
    baseline_progress: float = 0.0
    baseline_target: int = 0
    subgroups: int = 0
    spc_status: str = "BASELINING"
    capability_window: int = 0

    limits: ChartLimits | None = None
    chart_rows: tuple[dict[str, Any], ...] = ()
    capability: Capability | None = None
    capability_rows: tuple[dict[str, Any], ...] = ()
    active_alarms: tuple[Alarm, ...] = ()
    alarm_history: tuple[Alarm, ...] = ()

    @property
    def is_baselining(self) -> bool:
        """True while the engine is still collecting its baseline."""
        return self.phase == "baseline"

    @property
    def has_chart(self) -> bool:
        """True once there is something to draw."""
        return self.limits is not None and bool(self.chart_rows)


class LiveSource:
    """Runs the OPC UA client and the SPC engine in a background thread.

    Example:
        >>> source = LiveSource(scenario="none", speed=50.0)
        >>> source.snapshot().connected
        False
    """

    def __init__(
        self,
        scenario: str = "tool-wear",
        speed: float = 30.0,
        endpoint: str | None = None,
        baseline_subgroups: int = 25,
        capability_window: int = 15,
        rules: Sequence[int] | None = None,
        config: MachineConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Build a source.

        Args:
            scenario: Fault preset for the internal server. Ignored when
                connecting to an external endpoint.
            speed: Simulated seconds per wall-clock second, so a shift of
                production can be watched over a coffee.
            endpoint: An existing OPC UA server to connect to. When omitted,
                one is started inside this process.
            baseline_subgroups: How many subgroups before limits are frozen.
            capability_window: How many subgroups each rolling Cpk uses. Fewer
                means the trend starts sooner and each point is noisier.
            rules: Which Nelson Rules to apply. All eight if omitted.
            config: The machine definition.
            seed: Random seed override.
        """
        if scenario not in FAULT_PRESETS:
            raise ValueError(
                f"Unknown scenario {scenario!r}. Choose from {sorted(FAULT_PRESETS)}."
            )
        self.config = config if config is not None else load_config()
        self.scenario = scenario
        self.speed = speed
        self.endpoint = endpoint
        self.baseline_subgroups = baseline_subgroups
        self.capability_window = capability_window
        self.rules = tuple(rules) if rules is not None else None
        self.seed = seed

        self._lock = threading.Lock()
        self._snapshot = Snapshot(scenario=scenario, speed=speed)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._engine: SPCEngine | None = None
        self._started_at: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """True while the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background thread. Does nothing if already running."""
        if self.running:
            return
        self._stop.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._thread_main, name="spc-live-source", daemon=True
        )
        self._thread.start()
        logger.info(
            "Live source started (scenario %s, speed x%g)", self.scenario, self.speed
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the background thread to shut down and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("Live source stopped")

    def snapshot(self) -> Snapshot:
        """Return the current frozen view. Safe to call from any thread."""
        with self._lock:
            return self._snapshot

    def acknowledge_alarms(self) -> int:
        """Mark every active alarm as seen. Safe to call from the interface."""
        with self._lock:
            if self._engine is None:
                return 0
            count = self._engine.alarms.acknowledge_all()
            self._snapshot = _with_alarms(self._snapshot, self._engine)
            return count

    def wait_until(self, predicate, timeout: float = 30.0) -> bool:
        """Block until a snapshot satisfies a condition. For tests and scripts.

        Args:
            predicate: Called with each snapshot; return True to stop waiting.
            timeout: Seconds to give up after.

        Returns:
            True if the condition was met, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self.snapshot()):
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    # The background thread
    # ------------------------------------------------------------------

    def _thread_main(self) -> None:
        """Entry point for the background thread: one event loop, one job."""
        try:
            asyncio.run(self._run())
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Live source failed")
            with self._lock:
                self._snapshot = _replace(
                    self._snapshot, connected=False, error=str(exc)
                )

    async def _run(self) -> None:
        """Own the server, the client, the collector and the engine."""
        server: MachineServer | None = None
        publisher: asyncio.Task | None = None
        endpoint = self.endpoint

        if endpoint is None:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = int(probe.getsockname()[1])
            schedule = FaultSchedule(
                FAULT_PRESETS[self.scenario],
                seed=self.seed or self.config.random_seed,
            )
            simulator = MachineSimulator(self.config, seed=self.seed, faults=schedule)
            server = MachineServer(
                self.config,
                ServerSettings(port=port, speed_factor=self.speed),
                simulator,
            )
            await server.start()
            publisher = asyncio.create_task(server.run())
            endpoint = server.endpoint

        collector = DataCollector(self.config)
        engine = SPCEngine(
            self.config,
            rules=self.rules,
            baseline_subgroups=self.baseline_subgroups,
            capability_window=self.capability_window,
        )
        with self._lock:
            self._engine = engine

        try:
            async with MachineClient(ClientSettings(endpoint=endpoint)) as client:
                await client.subscribe()
                with self._lock:
                    self._snapshot = _replace(
                        self._snapshot, connected=True, endpoint=endpoint, error=None
                    )
                await self._pump(client, collector, engine)
        except Exception as exc:
            logger.exception("Live source connection failed")
            with self._lock:
                self._snapshot = _replace(
                    self._snapshot, connected=False, error=str(exc)
                )
        finally:
            if publisher is not None:
                publisher.cancel()
                try:
                    await publisher
                except asyncio.CancelledError:
                    pass
            if server is not None:
                await server.stop()
            with self._lock:
                self._snapshot = _replace(self._snapshot, connected=False)

    async def _pump(
        self, client: MachineClient, collector: DataCollector, engine: SPCEngine
    ) -> None:
        """Drain the client into the collector and the engine until asked to stop."""
        last_refresh = 0.0
        while not self._stop.is_set():
            drew_something = False
            async for update in client.updates(limit=1, timeout_s=0.2):
                drew_something = True
                subgroup = collector.handle(update)
                if subgroup is not None:
                    engine.add(subgroup)
                    self._refresh(client, collector, engine, full=True)
                    last_refresh = time.monotonic()

            now = time.monotonic()
            if not drew_something or now - last_refresh >= SNAPSHOT_INTERVAL_S:
                self._refresh(client, collector, engine, full=False)
                last_refresh = now

    def _refresh(
        self,
        client: MachineClient,
        collector: DataCollector,
        engine: SPCEngine,
        full: bool,
    ) -> None:
        """Rebuild the snapshot. Called only from the background thread."""
        latest = dict(collector.latest)
        ages = {name: client.age_s(name) for name in latest}

        with self._lock:
            previous = self._snapshot
            updated = _replace(
                previous,
                connected=True,
                machine_status=str(latest.get("Status", "UNKNOWN")),
                latest=latest,
                ages=ages,
                parts=len(collector.parts),
                recent_parts=tuple(
                    record.value for record in collector.parts[-RECENT_PARTS:]
                ),
                scrap=int(latest.get("ScrapCount", 0) or 0),
                missed=collector.missed_parts,
                updates=collector.updates_received,
                phase=engine.phase,
                baseline_progress=engine.baseline_progress,
                baseline_target=engine.baseline_subgroups,
                subgroups=engine.subgroups_monitored,
                spc_status=engine.status,
                capability_window=engine.capability_window,
            )
            if full:
                updated = _replace(
                    updated,
                    limits=engine.limits,
                    chart_rows=tuple(point.as_row() for point in engine.points),
                    capability=engine.capability,
                    capability_rows=tuple(
                        dict(
                            capability.as_row(),
                            subgroup=i + engine.capability_window - 1,
                        )
                        for i, capability in enumerate(engine.capability_trend)
                    ),
                )
                updated = _with_alarms(updated, engine)
            self._snapshot = updated


def _replace(snapshot: Snapshot, **changes: Any) -> Snapshot:
    """dataclasses.replace, kept local so the import list stays short."""
    from dataclasses import replace

    return replace(snapshot, **changes)


def _with_alarms(snapshot: Snapshot, engine: SPCEngine) -> Snapshot:
    """Copy the engine's alarm lists into a snapshot."""
    return _replace(
        snapshot,
        active_alarms=engine.alarms.active,
        alarm_history=engine.alarms.history,
        spc_status=engine.status,
    )