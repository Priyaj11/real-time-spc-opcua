"""Small statistical helpers used by the machine simulator.

These are deliberately thin wrappers around NumPy. Keeping them here, in one
place, means every random number in the project comes from a single seeded
generator, which is what makes runs reproducible.
"""

from __future__ import annotations

import math

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a random number generator that always produces the same sequence.

    Args:
        seed: Any integer. The same seed always gives the same stream of numbers.

    Returns:
        A NumPy Generator.

    Note:
        We use numpy.random.default_rng rather than the older numpy.random.seed
        because default_rng gives an isolated generator. Two simulators built
        with the same seed do not interfere with each other, and nothing else in
        the process can accidentally consume our random numbers.
    """
    return np.random.default_rng(seed)


def normal(rng: np.random.Generator, mean: float, std_dev: float) -> float:
    """Draw one value from a normal (bell curve) distribution.

    Args:
        rng: The seeded generator.
        mean: Centre of the bell curve.
        std_dev: Spread. Roughly 68 percent of values land within one std_dev of
            the mean, and about 99.7 percent within three.

    Returns:
        A single float.
    """
    if std_dev <= 0.0:
        return mean
    return float(rng.normal(loc=mean, scale=std_dev))


def exponential_approach(
    elapsed_s: float, start: float, target: float, tau_s: float
) -> float:
    """Value of a quantity warming up (or cooling down) toward a steady state.

    This is the standard first-order response you see in any thermal system:
    fast change at first, then slower as it closes on the target.

        value(t) = target + (start - target) * exp(-t / tau)

    Args:
        elapsed_s: Seconds since the process started.
        start: Value at time zero, for example ambient temperature.
        target: Value it settles at after a long time.
        tau_s: Time constant in seconds. After one tau the gap has closed by
            about 63 percent, after three tau by about 95 percent.

    Returns:
        The value at elapsed_s.
    """
    if tau_s <= 0.0:
        return target
    return target + (start - target) * math.exp(-elapsed_s / tau_s)


def split_variance(total_std_dev: float, explained_std_dev: float) -> float:
    """Work out how much independent noise to add when part of the spread is explained.

    If a tag's total spread is known, and some of that spread already comes from
    another tag it is coupled to, the leftover independent noise must be smaller
    so the total still comes out right.

    Variances add, standard deviations do not:

        total_variance = explained_variance + independent_variance

    Args:
        total_std_dev: The spread the tag should have overall.
        explained_std_dev: The spread already contributed by the coupled tag.

    Returns:
        The standard deviation of the remaining independent noise, never negative.
    """
    total_var = total_std_dev**2
    explained_var = explained_std_dev**2
    remaining = total_var - explained_var
    return math.sqrt(remaining) if remaining > 0.0 else 0.0