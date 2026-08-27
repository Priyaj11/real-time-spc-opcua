"""A simulated CNC boring station that behaves like a real machine.

Two kinds of tag, updated on different clocks, because that is how a real
machine works:

  Continuous tags (Torque, Temperature, Vibration)
      Sampled by the controller every 100 ms. New value on every step.

  Per-part tags (BoreDiameter, CycleTime, ScrapCount)
      A bore is measured once, when the part finishes. Between parts the
      machine holds the last measured value, exactly as an OPC UA server does.

Every sample carries two dictionaries. `values` is what the machine publishes,
which a sensor fault can corrupt. `truth` is the physical reality, which only
a process fault can change. Scrap is judged on the truth, never on the reading.

The simulator has no concept of wall-clock time and never sleeps. It advances
one sample per call to step(). Real-time pacing belongs to the OPC UA server in
Milestone 4, and tests and evaluation runs go as fast as the processor allows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from spc_opcua.config import MachineConfig, TagSpec, load_config
from spc_opcua.simulator.distributions import (
    exponential_approach,
    make_rng,
    normal,
    split_variance,
)
from spc_opcua.simulator.faults import FaultSchedule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulatorSettings:
    """Tuning knobs for the simulation itself, as opposed to the machine spec.

    These describe how we choose to model the machine. They are not customer
    requirements, which is why they live here and not in machine.yaml.

    Attributes:
        ambient_temp_c: Shop floor temperature the spindle starts at.
        thermal_tau_s: Thermal time constant. Larger means slower warm-up.
        cold_start: Start cold and warm up, or start already at steady state.
        torque_vibration_coupling: Extra vibration in mm/s per Nm of torque
            above nominal. A harder cut shakes more.
        scrap_check_enabled: Count a part as scrap when its true bore falls
            outside the specification limits.
    """

    ambient_temp_c: float = 22.0
    thermal_tau_s: float = 180.0
    cold_start: bool = True
    torque_vibration_coupling: float = 0.08
    scrap_check_enabled: bool = True


@dataclass(frozen=True)
class Sample:
    """One snapshot of every machine tag at one instant.

    Attributes:
        index: Sample number since the run started, counting from zero.
        t_s: Simulated seconds since the run started.
        values: What the machine publishes. Sensor faults corrupt these.
        truth: The physical reality behind each reading. Identical to values
            on a healthy machine and under any process fault.
        part_completed: True on the sample where a part finished, which is the
            only sample carrying a fresh BoreDiameter measurement.
        part_index: How many parts have been completed so far.
        active_faults: Names of every fault running at this instant.
    """

    index: int
    t_s: float
    values: dict[str, float] = field(default_factory=dict)
    truth: dict[str, float] = field(default_factory=dict)
    part_completed: bool = False
    part_index: int = 0
    active_faults: tuple[str, ...] = ()

    def __getitem__(self, tag_name: str) -> float:
        """Allow sample["Torque"] as a shorthand for sample.values["Torque"]."""
        return self.values[tag_name]

    @property
    def is_faulted(self) -> bool:
        """True when at least one fault is running at this instant."""
        return bool(self.active_faults)


class MachineSimulator:
    """Generates realistic process data for the simulated boring station.

    Example:
        >>> from spc_opcua.config import load_config
        >>> sim = MachineSimulator(load_config(), seed=42)
        >>> first = sim.step()
        >>> round(first.t_s, 3)
        0.0
    """

    def __init__(
        self,
        config: MachineConfig | None = None,
        seed: int | None = None,
        settings: SimulatorSettings | None = None,
        faults: FaultSchedule | None = None,
    ) -> None:
        """Build a simulator.

        Args:
            config: The machine definition. Loaded from machine.yaml if omitted.
            seed: Overrides the seed in the configuration. Same seed, same data.
            settings: Simulation tuning. Sensible defaults if omitted.
            faults: What is going wrong, and when. Healthy machine if omitted.
        """
        self.config = config if config is not None else load_config()
        self.settings = settings if settings is not None else SimulatorSettings()
        self.seed = seed if seed is not None else self.config.random_seed
        self.faults = faults if faults is not None else FaultSchedule()

        self._bore = self.config.tag("BoreDiameter")
        self._torque = self.config.tag("Torque")
        self._cycle = self.config.tag("CycleTime")
        self._temp = self.config.tag("Temperature")
        self._vib = self.config.tag("Vibration")

        # How much of vibration's spread the torque coupling already explains,
        # so the independent noise can be reduced to keep the total correct.
        coupled_vib_std = abs(self.settings.torque_vibration_coupling) * (
            self._torque.std_dev
        )
        self._vib_independent_std = split_variance(self._vib.std_dev, coupled_vib_std)

        self._rng: np.random.Generator
        self.reset()

        logger.info(
            "Simulator ready for %s, seed %d, %.1f Hz, %d fault(s)",
            self.config.name,
            self.seed,
            self.config.sample_rate_hz,
            len(self.faults),
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return to time zero with fresh generators, reproducing the same run."""
        self._rng = make_rng(self.seed)
        self.faults.reset()
        self._sample_index = 0
        self._t_s = 0.0
        self._parts_completed = 0
        self._scrap_count = 0.0
        self._last_published: dict[str, float] = {}

        # Per-part tags hold their last value between parts, so seed them with a
        # plausible first reading rather than leaving them at zero.
        self._last_bore = normal(self._rng, self._bore.nominal, self._bore.std_dev)
        self._last_cycle_time = normal(
            self._rng, self._cycle.nominal, self._cycle.std_dev
        )
        self._next_part_t_s = self._last_cycle_time

        start_temp = (
            self.settings.ambient_temp_c
            if self.settings.cold_start
            else self._temp.nominal
        )
        self._temp_start_c = start_temp

    @property
    def elapsed_s(self) -> float:
        """Simulated seconds produced so far."""
        return self._t_s

    @property
    def parts_completed(self) -> int:
        """Number of finished parts so far."""
        return self._parts_completed

    @property
    def scrap_count(self) -> int:
        """Number of parts rejected so far, judged on the true bore size."""
        return int(self._scrap_count)

    # ------------------------------------------------------------------
    # Drawing one true value, with any process fault applied
    # ------------------------------------------------------------------

    def _draw(
        self, spec: TagSpec, mean: float, std_override: float | None = None
    ) -> float:
        """Draw one true value for a tag, applying any active process fault.

        Args:
            spec: The tag being drawn.
            mean: The healthy centre for this draw. For Temperature this is the
                warm-up curve, not the nominal.
            std_override: Use this spread instead of the tag's configured one.
                Vibration needs it, because part of its spread comes from torque.

        Returns:
            The physically true value, before any sensor fault.
        """
        effect = self.faults.process_effect(self._t_s, spec)
        base_std = spec.std_dev if std_override is None else std_override
        value = normal(
            self._rng,
            mean + effect.mean_offset,
            base_std * effect.std_multiplier,
        )
        return value + effect.spike

    def _draw_torque(self) -> float:
        """Spindle torque. Independent noise around nominal."""
        return self._draw(self._torque, self._torque.nominal)

    def _draw_temperature(self) -> float:
        """Spindle temperature: a warm-up curve plus measurement noise."""
        baseline = exponential_approach(
            elapsed_s=self._t_s,
            start=self._temp_start_c,
            target=self._temp.nominal,
            tau_s=self.settings.thermal_tau_s,
        )
        return self._draw(self._temp, baseline)

    def _draw_vibration(self, torque: float) -> float:
        """Vibration: partly driven by torque, partly independent noise."""
        torque_excess = torque - self._torque.nominal
        coupled = self.settings.torque_vibration_coupling * torque_excess
        return self._draw(
            self._vib,
            self._vib.nominal + coupled,
            std_override=self._vib_independent_std,
        )

    def _draw_bore_diameter(self) -> float:
        """True finished bore diameter for one part."""
        return self._draw(self._bore, self._bore.nominal)

    def _draw_cycle_time(self) -> float:
        """Time this part took. Never allowed to go negative."""
        return max(self._draw(self._cycle, self._cycle.nominal), 0.1)

    def _is_scrap(self, true_bore: float) -> bool:
        """True when the real part falls outside the specification limits."""
        if not self.settings.scrap_check_enabled:
            return False
        if self._bore.lsl is not None and true_bore < self._bore.lsl:
            return True
        if self._bore.usl is not None and true_bore > self._bore.usl:
            return True
        return False

    # ------------------------------------------------------------------
    # Turning a true value into a published reading
    # ------------------------------------------------------------------

    def _publish(self, spec: TagSpec, true_value: float) -> float:
        """Apply any active sensor fault to produce the reported value.

        A stuck sensor repeats whatever it last reported. Otherwise the reading
        is the truth plus a calibration offset plus any extra gauge noise.
        """
        effect = self.faults.sensor_effect(self._t_s, spec)
        if effect.is_neutral:
            self._last_published[spec.name] = true_value
            return true_value
        if effect.frozen:
            # setdefault, not get: the first frozen sample becomes the value the
            # sensor repeats forever after.
            return self._last_published.setdefault(spec.name, true_value)
        reading = (
            true_value
            + effect.offset
            + self.faults.sensor_noise(effect.extra_noise_std)
        )
        self._last_published[spec.name] = reading
        return reading

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self) -> Sample:
        """Advance the simulation by one sample period and return the new sample.

        Returns:
            A Sample holding a published value and a true value for every tag.
        """
        torque_true = self._draw_torque()
        temperature_true = self._draw_temperature()
        vibration_true = self._draw_vibration(torque_true)

        part_completed = False
        if self._t_s >= self._next_part_t_s:
            part_completed = True
            self._parts_completed += 1
            self._last_bore = self._draw_bore_diameter()
            self._last_cycle_time = self._draw_cycle_time()
            self._next_part_t_s = self._t_s + self._last_cycle_time
            if self._is_scrap(self._last_bore):
                self._scrap_count += 1.0
                logger.debug(
                    "Part %d scrapped, true bore %.4f mm at t=%.1f s",
                    self._parts_completed,
                    self._last_bore,
                    self._t_s,
                )

        truth = {
            "BoreDiameter": self._last_bore,
            "Torque": torque_true,
            "CycleTime": self._last_cycle_time,
            "Temperature": temperature_true,
            "Vibration": vibration_true,
            "ScrapCount": self._scrap_count,
        }
        values = {
            "BoreDiameter": self._publish(self._bore, self._last_bore),
            "Torque": self._publish(self._torque, torque_true),
            "CycleTime": self._publish(self._cycle, self._last_cycle_time),
            "Temperature": self._publish(self._temp, temperature_true),
            "Vibration": self._publish(self._vib, vibration_true),
            "ScrapCount": self._scrap_count,
        }

        sample = Sample(
            index=self._sample_index,
            t_s=self._t_s,
            values=values,
            truth=truth,
            part_completed=part_completed,
            part_index=self._parts_completed,
            active_faults=tuple(self.faults.labels_at(self._t_s)),
        )

        self._sample_index += 1
        self._t_s += self.config.sample_period_s
        return sample

    def run(self, n_samples: int) -> list[Sample]:
        """Advance the simulation n_samples times and return every sample."""
        if n_samples < 0:
            raise ValueError("n_samples must not be negative")
        return [self.step() for _ in range(n_samples)]

    def run_seconds(self, duration_s: float) -> list[Sample]:
        """Advance the simulation for a number of simulated seconds."""
        n = int(round(duration_s * self.config.sample_rate_hz))
        return self.run(n)

    def stream(self) -> Iterator[Sample]:
        """Yield samples forever. Use with care, this never stops on its own."""
        while True:
            yield self.step()


def main() -> None:
    """Compare a healthy hour with a tool-wear hour. Entry point for a smoke test."""
    from spc_opcua.logging_setup import configure_logging
    from spc_opcua.simulator.faults import ToolWear

    configure_logging()
    config = load_config()
    bore = config.tag("BoreDiameter")

    def summarise(title: str, sim: MachineSimulator) -> None:
        samples = sim.run_seconds(3600.0)
        bores = [s.truth["BoreDiameter"] for s in samples if s.part_completed]
        first_ten = float(np.mean(bores[:10]))
        last_ten = float(np.mean(bores[-10:]))
        print(f"\n{title}")
        print(f"  parts completed   {sim.parts_completed}")
        print(f"  parts scrapped    {sim.scrap_count}")
        print(f"  mean bore         {float(np.mean(bores)):.5f} mm")
        print(f"  first 10 parts    {first_ten:.5f} mm")
        print(f"  last 10 parts     {last_ten:.5f} mm")
        print(f"  drift             {last_ten - first_ten:+.5f} mm")

    summarise("HEALTHY", MachineSimulator(config, seed=42))

    wear = FaultSchedule(
        [
            ToolWear(tag="BoreDiameter", start_s=600.0, rate_per_hour=-0.045),
            ToolWear(tag="Torque", start_s=600.0, rate_per_hour=3.0),
        ],
        seed=42,
    )
    summarise("TOOL WEAR from t=600 s", MachineSimulator(config, seed=42, faults=wear))

    print(f"\nSpecification limits: {bore.lsl:.3f} to {bore.usl:.3f} mm")
    print()


if __name__ == "__main__":
    main()