"""Subgroups: the unit of Statistical Process Control.

SPC does not plot every measurement. It gathers a handful of consecutive parts,
five in this project, and plots one point per handful. That handful is a
subgroup.

Two reasons it works better than plotting raw measurements:

  Averages are steadier than single values.
      The average of five parts wobbles about 2.2 times less than a single
      part, because spread shrinks with the square root of the count. A real
      shift stands out against a quieter background.

  A subgroup measures two different things at once.
      Its MEAN says where the process is centred. Its RANGE, largest minus
      smallest, says how consistent it is. A machine can drift off centre
      while staying consistent, or stay perfectly centred while becoming
      erratic. You need both numbers to tell those apart, which is why
      Milestone 7 builds two charts rather than one.

The one rule that matters: a subgroup must be consecutive parts produced under
the same conditions. Five parts in a row, not five parts picked from a shift.
The whole method rests on the variation WITHIN a subgroup being pure short-term
noise, so that variation BETWEEN subgroups is the signal.

Nothing in this module knows about OPC UA, networks, or the simulator. It takes
numbers and groups them.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

MIN_SUBGROUP_SIZE = 2
MAX_SUBGROUP_SIZE = 25  # beyond this there are no standard SPC constants


@dataclass(frozen=True)
class Subgroup:
    """One batch of consecutive measurements, ready to become a chart point.

    Attributes:
        index: Which subgroup this is, counting from zero.
        tag: Which measurement these values are.
        values: The measurements themselves, in production order.
        part_indices: Which parts they came from, for traceability back to the
            machine. Empty when the subgroup was built from bare numbers.
        timestamps: When each part was measured, where known.
    """

    index: int
    tag: str
    values: tuple[float, ...]
    part_indices: tuple[int, ...] = ()
    timestamps: tuple[datetime | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("A subgroup must contain at least one value")
        if self.part_indices and len(self.part_indices) != len(self.values):
            raise ValueError("part_indices must match the number of values")
        if self.timestamps and len(self.timestamps) != len(self.values):
            raise ValueError("timestamps must match the number of values")

    @property
    def size(self) -> int:
        """How many parts are in this subgroup. Called n in every SPC textbook."""
        return len(self.values)

    @property
    def mean(self) -> float:
        """The average. Plotted on the X-bar chart, and written x-bar."""
        return sum(self.values) / len(self.values)

    @property
    def range(self) -> float:
        """Largest minus smallest. Plotted on the R chart.

        Range rather than standard deviation because it is trivial to compute
        by hand on a shop floor, and for small subgroups it is very nearly as
        good. That is a 1920s decision the standard never had reason to revisit.
        """
        return max(self.values) - min(self.values)

    @property
    def minimum(self) -> float:
        """Smallest value in the subgroup."""
        return min(self.values)

    @property
    def maximum(self) -> float:
        """Largest value in the subgroup."""
        return max(self.values)

    @property
    def std_dev(self) -> float:
        """Sample standard deviation within the subgroup.

        Not used for the control limits, which are built from the range, but
        useful for reporting and for checking the range-based estimate.
        """
        if self.size < 2:
            return 0.0
        average = self.mean
        variance = sum((v - average) ** 2 for v in self.values) / (self.size - 1)
        return math.sqrt(variance)

    @property
    def first_timestamp(self) -> datetime | None:
        """When the first part in this subgroup was measured, if known."""
        return self.timestamps[0] if self.timestamps else None

    @property
    def last_timestamp(self) -> datetime | None:
        """When the last part in this subgroup was measured, if known."""
        return self.timestamps[-1] if self.timestamps else None

    def as_row(self) -> dict[str, object]:
        """Flatten into one dictionary, suitable for a table or a CSV row."""
        return {
            "subgroup": self.index,
            "tag": self.tag,
            "n": self.size,
            "mean": self.mean,
            "range": self.range,
            "min": self.minimum,
            "max": self.maximum,
            "std_dev": self.std_dev,
            "first_part": self.part_indices[0] if self.part_indices else None,
            "last_part": self.part_indices[-1] if self.part_indices else None,
            "timestamp": self.last_timestamp,
        }


class SubgroupBuilder:
    """Collects measurements one at a time and emits a Subgroup every n of them.

    Example:
        >>> builder = SubgroupBuilder(tag="BoreDiameter", size=2)
        >>> builder.add(20.01) is None
        True
        >>> group = builder.add(19.99)
        >>> group.size, round(group.mean, 4), round(group.range, 4)
        (2, 20.0, 0.02)
    """

    def __init__(self, tag: str, size: int) -> None:
        """Build a subgroup builder.

        Args:
            tag: Which measurement this builder collects.
            size: How many consecutive parts form one subgroup.

        Raises:
            ValueError: if the size has no standard SPC constants.
        """
        if not tag:
            raise ValueError("A subgroup builder needs a tag name")
        if size < MIN_SUBGROUP_SIZE:
            raise ValueError(f"Subgroup size must be at least {MIN_SUBGROUP_SIZE}")
        if size > MAX_SUBGROUP_SIZE:
            raise ValueError(
                f"Subgroup size above {MAX_SUBGROUP_SIZE} has no standard constants"
            )

        self.tag = tag
        self.size = size
        self._values: list[float] = []
        self._parts: list[int] = []
        self._times: list[datetime | None] = []
        self._emitted = 0

    def __len__(self) -> int:
        """How many measurements are waiting for the subgroup to fill."""
        return len(self._values)

    @property
    def pending(self) -> tuple[float, ...]:
        """Measurements collected so far towards the next subgroup."""
        return tuple(self._values)

    @property
    def subgroups_emitted(self) -> int:
        """How many complete subgroups this builder has produced."""
        return self._emitted

    def reset(self) -> None:
        """Throw away the partial subgroup and the emitted count."""
        self._values.clear()
        self._parts.clear()
        self._times.clear()
        self._emitted = 0

    def discard_partial(self) -> None:
        """Throw away the partial subgroup, keeping the emitted count.

        Use this when production restarts after a stoppage. Parts from either
        side of a break are not consecutive under the same conditions, so
        mixing them into one subgroup would inflate its range and widen the
        control limits for every later point.
        """
        if self._values:
            logger.debug(
                "Discarding %d part(s) of a partial subgroup", len(self._values)
            )
        self._values.clear()
        self._parts.clear()
        self._times.clear()

    def add(
        self,
        value: float,
        part_index: int | None = None,
        timestamp: datetime | None = None,
    ) -> Subgroup | None:
        """Add one measurement.

        Args:
            value: The measurement.
            part_index: Which part it came from, for traceability.
            timestamp: When it was measured.

        Returns:
            The completed Subgroup if this measurement filled one, else None.
        """
        self._values.append(float(value))
        self._parts.append(-1 if part_index is None else int(part_index))
        self._times.append(timestamp)

        if len(self._values) < self.size:
            return None

        subgroup = Subgroup(
            index=self._emitted,
            tag=self.tag,
            values=tuple(self._values),
            part_indices=tuple(self._parts),
            timestamps=tuple(self._times),
        )
        self._emitted += 1
        self._values.clear()
        self._parts.clear()
        self._times.clear()
        return subgroup

    def extend(self, values: Iterable[float]) -> list[Subgroup]:
        """Add many measurements and return every subgroup they completed."""
        completed = []
        for value in values:
            subgroup = self.add(value)
            if subgroup is not None:
                completed.append(subgroup)
        return completed


def subgroups_from_values(
    values: Sequence[float], size: int, tag: str = "value"
) -> list[Subgroup]:
    """Split a list of measurements into subgroups, dropping any short remainder.

    Handy for offline analysis of a saved data file, which is how Milestone 7
    develops the control charts before wiring them to the live stream.

    Args:
        values: Measurements in production order.
        size: Parts per subgroup.
        tag: Name to record on each subgroup.

    Returns:
        Complete subgroups only. A trailing partial batch is dropped, because a
        subgroup of three plotted next to subgroups of five would have a
        systematically smaller range and a misleading control limit.
    """
    builder = SubgroupBuilder(tag=tag, size=size)
    return builder.extend(values)