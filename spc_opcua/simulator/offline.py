"""Run the simulator without a clock, a server or a network.

The live path is Milestone 5's: the server paces itself in real time and pushes
values to subscribed clients. That is the right shape for a dashboard and the
wrong shape for a study. Answering "how many subgroups before this fault is
detected" means running the same fault hundreds of times, and nobody waits four
hours of wall-clock time for that.

So this module drives MachineSimulator.step() as fast as the processor allows
and collects the results. Same simulator, same faults, same seeds, same
numbers as the live path would eventually produce; only the pacing is gone.

Everything here is cached on its arguments. The simulator is deterministic for
a given seed, which Milestone 2 proved with a test, so the same request always
yields the same values and there is no reason to compute them twice. Results
are returned as tuples so a caller cannot mutate what the next caller receives.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from spc_opcua.config import MachineConfig, load_config
from spc_opcua.simulator.faults import Fault, FaultSchedule
from spc_opcua.simulator.machine import MachineSimulator
from spc_opcua.spc.subgroups import Subgroup, subgroups_from_values

DEFAULT_TAG = "BoreDiameter"

# How many cached runs to keep. Each entry is a tuple of floats, so a few
# hundred parts is a few kilobytes; 128 of them costs nothing worth measuring.
CACHE_SIZE = 128


@lru_cache(maxsize=CACHE_SIZE)
def part_values(
    faults: tuple[Fault, ...] = (),
    seed: int = 1,
    parts: int = 200,
    tag: str = DEFAULT_TAG,
    fault_seed: int = 1,
) -> tuple[float, ...]:
    """Measure one characteristic on each of the next N completed parts.

    Args:
        faults: Faults to inject. Empty means a healthy machine.
        seed: Seed for the machine's own noise.
        parts: How many completed parts to collect.
        tag: Which measurement to keep.
        fault_seed: Seed for the fault schedule's separate generator.

    Returns:
        One value per part, in production order.

    Example:
        >>> values = part_values(parts=5)
        >>> len(values)
        5
    """
    config = load_config()
    schedule = FaultSchedule(faults, seed=fault_seed)
    simulator = MachineSimulator(config, seed=seed, faults=schedule)
    collected: list[float] = []
    while len(collected) < parts:
        sample = simulator.step()
        if sample.part_completed:
            collected.append(sample.values[tag])
    return tuple(collected)


def bore_subgroups(
    faults: FaultSchedule | Sequence[Fault] = (),
    seed: int = 1,
    count: int = 25,
    config: MachineConfig | None = None,
    tag: str = DEFAULT_TAG,
) -> list[Subgroup]:
    """Produce N subgroups of bore measurements from an offline run.

    Args:
        faults: A FaultSchedule, or a plain sequence of faults.
        seed: Seed for the machine's noise.
        count: How many subgroups to return.
        config: Machine definition. Loaded from machine.yaml if omitted.
        tag: Which measurement to chart.

    Returns:
        A list of Subgroup, numbered from zero.

    Example:
        >>> groups = bore_subgroups(count=3)
        >>> len(groups)
        3
    """
    config = config if config is not None else load_config()
    size = config.subgroup_size

    if isinstance(faults, FaultSchedule):
        fault_tuple, fault_seed = faults.faults, faults.seed
    else:
        fault_tuple, fault_seed = tuple(faults), 1

    values = part_values(
        fault_tuple, seed=seed, parts=count * size, tag=tag, fault_seed=fault_seed
    )
    return subgroups_from_values(list(values), size, tag=tag)