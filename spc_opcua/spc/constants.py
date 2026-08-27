"""The SPC control chart constants, and where they come from.

Every SPC textbook prints a table of constants called A2, D3 and D4, and most
people copy them without ever being told what they are. They are not magic.
They all fall out of two numbers, d2 and d3, which describe how the RANGE of a
small sample behaves when the underlying process is normal.

d2, the mean of the relative range
    Take n values from a normal process with standard deviation sigma. The
    average range you get is d2 times sigma. So if you measure an average range
    and divide by d2, you get an estimate of sigma. For n = 5, d2 is 2.326.

d3, the standard deviation of the relative range
    Ranges themselves vary. Their standard deviation is d3 times sigma.

From those two, the printed constants follow by simple arithmetic:

    A2 = 3 / (d2 * sqrt(n))
        The X-bar chart plots subgroup MEANS. The standard deviation of a mean
        of n values is sigma / sqrt(n), so three of those is
        3 * sigma / sqrt(n). Substituting sigma = R-bar / d2 gives
        3 * R-bar / (d2 * sqrt(n)), which is A2 times R-bar.

    D4 = 1 + 3 * d3 / d2      D3 = 1 - 3 * d3 / d2, floored at zero
        The R chart plots RANGES. Their centre is R-bar and their standard
        deviation is d3 * sigma = d3 * R-bar / d2. Three of those either side
        of R-bar gives the two constants. For n of 6 or less, D3 comes out
        negative, and a range cannot be negative, so the lower limit is zero.

This module tabulates d2 and d3 and derives the rest, rather than copying four
columns of a printed table. There are tests asserting the derived values match
the published ones, which is a self-check that no digit was mistyped.

Everything here assumes the underlying measurements are roughly normal. That
assumption is why an SPC engineer checks a histogram before trusting a chart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Subgroup size to (d2, d3). Standard values, as printed in ASTM and the AIAG
# SPC reference manual.
_RELATIVE_RANGE: dict[int, tuple[float, float]] = {
    2: (1.128, 0.8525),
    3: (1.693, 0.8884),
    4: (2.059, 0.8798),
    5: (2.326, 0.8641),
    6: (2.534, 0.8480),
    7: (2.704, 0.8332),
    8: (2.847, 0.8198),
    9: (2.970, 0.8078),
    10: (3.078, 0.7971),
    11: (3.173, 0.7873),
    12: (3.258, 0.7785),
    13: (3.336, 0.7704),
    14: (3.407, 0.7630),
    15: (3.472, 0.7562),
    16: (3.532, 0.7499),
    17: (3.588, 0.7441),
    18: (3.640, 0.7386),
    19: (3.689, 0.7335),
    20: (3.735, 0.7287),
    21: (3.778, 0.7242),
    22: (3.819, 0.7199),
    23: (3.858, 0.7159),
    24: (3.895, 0.7121),
    25: (3.931, 0.7084),
}

MIN_SUBGROUP_SIZE = min(_RELATIVE_RANGE)
MAX_SUBGROUP_SIZE = max(_RELATIVE_RANGE)

# How many baseline subgroups before control limits are worth trusting. Not a
# law, but the number every SPC reference gives, and worth warning below.
RECOMMENDED_BASELINE_SUBGROUPS = 20


@dataclass(frozen=True)
class ChartConstants:
    """The constants for one subgroup size.

    Attributes:
        n: Subgroup size.
        d2: Average range of n normal values, in units of sigma.
        d3: Standard deviation of that range, in units of sigma.
        a2: Multiplier on R-bar giving three sigma of the subgroup mean.
        d3_lower: The D3 constant, lower control limit multiplier for ranges.
        d4_upper: The D4 constant, upper control limit multiplier for ranges.
    """

    n: int
    d2: float
    d3: float
    a2: float
    d3_lower: float
    d4_upper: float


def constants_for(n: int) -> ChartConstants:
    """Look up and derive every constant for a subgroup size.

    Args:
        n: Subgroup size, between 2 and 25.

    Returns:
        The constants for that size.

    Raises:
        ValueError: if there are no standard constants for that size.
    """
    try:
        d2, d3 = _RELATIVE_RANGE[n]
    except KeyError as exc:
        raise ValueError(
            f"No standard SPC constants for a subgroup of {n}. "
            f"Supported sizes are {MIN_SUBGROUP_SIZE} to {MAX_SUBGROUP_SIZE}."
        ) from exc

    three_d3_over_d2 = 3.0 * d3 / d2
    return ChartConstants(
        n=n,
        d2=d2,
        d3=d3,
        a2=3.0 / (d2 * math.sqrt(n)),
        # A range can never be negative, so a negative D3 becomes zero. For
        # n of 6 or less this always happens, which is why small subgroups have
        # no lower limit on the R chart.
        d3_lower=max(0.0, 1.0 - three_d3_over_d2),
        d4_upper=1.0 + three_d3_over_d2,
    )


def sigma_from_mean_range(mean_range: float, n: int) -> float:
    """Estimate the process standard deviation from the average range.

    This is the within-subgroup, short-term spread. It deliberately ignores any
    drift between subgroups, which is the whole point: it answers "how good
    could this process be if it stayed put", and the chart then asks whether it
    is staying put.

    Args:
        mean_range: R-bar, the average of every subgroup's range.
        n: Subgroup size.

    Returns:
        Estimated sigma.
    """
    if mean_range < 0.0:
        raise ValueError("A mean range cannot be negative")
    return mean_range / constants_for(n).d2