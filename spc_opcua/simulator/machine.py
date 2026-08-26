"""A simulated CNC boring station that behaves like a real machine.

Two kinds of tag, updated on different clocks, because that is how a real
machine works:

  Continuous tags (Torque, Temperature, Vibration)
      Sampled by the controller every 100 ms. New value on every step.

  Per-part tags (BoreDiameter, CycleTime, ScrapCount)
      A bore is measured once, when the part finishes. Between parts the
      machine holds the last measured value, exactly as an OPC UA server does.

The simulator has no concept of wall-clock time and never sleeps. It advances
one sample per call to step(). Real-time pacing belongs to the OPC UA server in
Milestone 4, and tests and evaluation runs go as fast as the processor allows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from spc_opcua.config import MachineConfig, load_config
from spc_opcua.simulator.distributions import (
    exponential_approach,
    make_rng,
    normal,
    split_variance,
)

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
        scrap_check_enabled: Count a part as scrap when its bore falls outside
            the specification limits.
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
        values: Tag name to value, for every tag in the configuration.
        part_completed: True on the sample where a part finished, which is the
            only sample carrying a fresh BoreDiameter measurement.
        part_index: How many parts have been completed so far.
    """

    index: int
    t_s: float
    values: dict[str, float] = field(default_factory=dict)
    part_completed: bool = False
    part_index: int = 0

    def __getitem__(self, tag_name: str) -> float:
        """Allow sample["Torque"] as a shorthand for sample.values["Torque"]."""
        return self.values[tag_name]


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
    ) -> None:
        """Build a simulator.

        Args:
            config: The machine definition. Loaded from machine.yaml if omitted.
            seed: Overrides the seed in the configuration. Same seed, same data.
            settings: Simulation tuning. Sensible defaults if omitted.
        """
        self.config = config if config is not None else load_config()
        self.settings = settings if settings is not None else SimulatorSettings()
        self.seed = seed if seed is not None else self.config.random_seed

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
            "Simulator ready for %s, seed %d, %.1f Hz",
            self.config.name,
            self.seed,
            self.config.sample_rate_hz,
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return to time zero with a fresh generator, reproducing the same run."""
        self._rng = make_rng(self.seed)
        self._sample_index = 0
        self._t_s = 0.0
        self._parts_completed = 0
        self._scrap_count = 0.0

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
        """Number of parts rejected so far."""
        return int(self._scrap_count)

    # ------------------------------------------------------------------
    # Value models, one per tag
    # ------------------------------------------------------------------

    def _draw_torque(self) -> float:
        """Spindle torque. Independent noise around nominal."""
        return normal(self._rng, self._torque.nominal, self._torque.std_dev)

    def _draw_temperature(self) -> float:
        """Spindle temperature: a warm-up curve plus measurement noise."""
        baseline = exponential_approach(
            elapsed_s=self._t_s,
            start=self._temp_start_c,
            target=self._temp.nominal,
            tau_s=self.settings.thermal_tau_s,
        )
        return normal(self._rng, baseline, self._temp.std_dev)

    def _draw_vibration(self, torque: float) -> float:
        """Vibration: partly driven by torque, partly independent noise."""
        torque_excess = torque - self._torque.nominal
        coupled = self.settings.torque_vibration_coupling * torque_excess
        return normal(
            self._rng, self._vib.nominal + coupled, self._vib_independent_std
        )

    def _draw_bore_diameter(self) -> float:
        """Finished bore diameter for one part."""
        return normal(self._rng, self._bore.nominal, self._bore.std_dev)

    def _draw_cycle_time(self) -> float:
        """Time this part took. Never allowed to go negative."""
        value = normal(self._rng, self._cycle.nominal, self._cycle.std_dev)
        return max(value, 0.1)

    def _is_scrap(self, bore: float) -> bool:
        """True when a bore measurement falls outside the specification limits."""
        if not self.settings.scrap_check_enabled:
            return False
        if self._bore.lsl is not None and bore < self._bore.lsl:
            return True
        if self._bore.usl is not None and bore > self._bore.usl:
            return True
        return False

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self) -> Sample:
        """Advance the simulation by one sample period and return the new sample.

        Returns:
            A Sample holding a value for every configured tag.
        """
        torque = self._draw_torque()
        temperature = self._draw_temperature()
        vibration = self._draw_vibration(torque)

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
                    "Part %d scrapped, bore %.4f mm outside %.3f to %.3f",
                    self._parts_completed,
                    self._last_bore,
                    self._bore.lsl if self._bore.lsl is not None else float("nan"),
                    self._bore.usl if self._bore.usl is not None else float("nan"),
                )

        sample = Sample(
            index=self._sample_index,
            t_s=self._t_s,
            values={
                "BoreDiameter": self._last_bore,
                "Torque": torque,
                "CycleTime": self._last_cycle_time,
                "Temperature": temperature,
                "Vibration": vibration,
                "ScrapCount": self._scrap_count,
            },
            part_completed=part_completed,
            part_index=self._parts_completed,
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
    """Run five simulated minutes and print a summary. Entry point for a smoke test."""
    from spc_opcua.logging_setup import configure_logging

    configure_logging()
    sim = MachineSimulator()
    samples = sim.run_seconds(300.0)

    print(f"\nGenerated {len(samples)} samples over {sim.elapsed_s:.1f} simulated s")
    print(f"Parts completed : {sim.parts_completed}")
    print(f"Parts scrapped  : {sim.scrap_count}")

    print("\nFirst five samples:")
    print(f"{'t_s':>8}{'Bore':>12}{'Torque':>10}{'Temp':>9}{'Vib':>9}{'Part':>6}")
    print("-" * 54)
    for s in samples[:5]:
        print(
            f"{s.t_s:>8.1f}{s['BoreDiameter']:>12.4f}{s['Torque']:>10.2f}"
            f"{s['Temperature']:>9.2f}{s['Vibration']:>9.3f}{s.part_index:>6}"
        )

    bores = [s["BoreDiameter"] for s in samples if s.part_completed]
    print(f"\nBoreDiameter over {len(bores)} finished parts")
    print(f"  mean      {float(np.mean(bores)):.5f} mm  (nominal 20.00000)")
    print(f"  std dev   {float(np.std(bores, ddof=1)):.5f} mm  (configured 0.01200)")
    print(f"  min, max  {min(bores):.4f}, {max(bores):.4f} mm")

    first_temp = samples[0]["Temperature"]
    last_temp = samples[-1]["Temperature"]
    print(f"\nTemperature warm-up: {first_temp:.1f} degC -> {last_temp:.1f} degC")
    print()


if __name__ == "__main__":
    main()