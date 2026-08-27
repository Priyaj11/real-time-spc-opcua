"""Tests for subgroup formation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from spc_opcua.spc.subgroups import (
    Subgroup,
    SubgroupBuilder,
    subgroups_from_values,
)


def make_subgroup(values: tuple[float, ...], index: int = 0) -> Subgroup:
    return Subgroup(index=index, tag="BoreDiameter", values=values)


# --------------------------------------------------------------------------
# The statistics of one subgroup
# --------------------------------------------------------------------------


def test_the_mean_is_the_average() -> None:
    assert make_subgroup((1.0, 2.0, 3.0, 4.0)).mean == pytest.approx(2.5)


def test_the_range_is_largest_minus_smallest() -> None:
    assert make_subgroup((20.01, 19.98, 20.00)).range == pytest.approx(0.03)


def test_minimum_and_maximum_are_reported() -> None:
    group = make_subgroup((5.0, 1.0, 3.0))
    assert group.minimum == 1.0
    assert group.maximum == 5.0


def test_a_subgroup_of_identical_values_has_zero_range() -> None:
    assert make_subgroup((7.0, 7.0, 7.0)).range == 0.0


def test_the_size_is_the_count_of_values() -> None:
    assert make_subgroup((1.0, 2.0, 3.0, 4.0, 5.0)).size == 5


def test_the_standard_deviation_uses_the_sample_formula() -> None:
    """Dividing by n minus 1, because a subgroup is a sample, not a population."""
    group = make_subgroup((2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0))
    assert group.std_dev == pytest.approx(2.13809, abs=1e-5)


def test_a_single_value_has_no_spread() -> None:
    assert make_subgroup((3.0,)).std_dev == 0.0
    assert make_subgroup((3.0,)).range == 0.0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_an_empty_subgroup_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        Subgroup(index=0, tag="X", values=())


def test_mismatched_part_indices_are_rejected() -> None:
    with pytest.raises(ValueError, match="part_indices"):
        Subgroup(index=0, tag="X", values=(1.0, 2.0), part_indices=(1,))


def test_mismatched_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="timestamps"):
        Subgroup(index=0, tag="X", values=(1.0, 2.0), timestamps=(None,))


# --------------------------------------------------------------------------
# Traceability
# --------------------------------------------------------------------------


def test_a_subgroup_remembers_which_parts_it_came_from() -> None:
    group = Subgroup(
        index=3, tag="BoreDiameter", values=(1.0, 2.0), part_indices=(41, 42)
    )
    row = group.as_row()
    assert row["first_part"] == 41
    assert row["last_part"] == 42


def test_a_subgroup_reports_its_first_and_last_timestamps() -> None:
    start = datetime(2026, 8, 26, 9, 0, 0)
    later = start + timedelta(seconds=48)
    group = Subgroup(index=0, tag="X", values=(1.0, 2.0), timestamps=(start, later))
    assert group.first_timestamp == start
    assert group.last_timestamp == later


def test_timestamps_are_none_when_not_supplied() -> None:
    group = make_subgroup((1.0, 2.0))
    assert group.first_timestamp is None
    assert group.last_timestamp is None


def test_as_row_carries_every_field_a_chart_needs() -> None:
    row = make_subgroup((1.0, 3.0), index=7).as_row()
    assert row["subgroup"] == 7
    assert row["n"] == 2
    assert row["mean"] == pytest.approx(2.0)
    assert row["range"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Building subgroups from a stream
# --------------------------------------------------------------------------


def test_nothing_comes_out_until_the_subgroup_is_full() -> None:
    builder = SubgroupBuilder(tag="BoreDiameter", size=3)
    assert builder.add(1.0) is None
    assert builder.add(2.0) is None
    group = builder.add(3.0)
    assert group is not None
    assert group.values == (1.0, 2.0, 3.0)


def test_subgroup_indices_count_up() -> None:
    builder = SubgroupBuilder(tag="X", size=2)
    groups = builder.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert [g.index for g in groups] == [0, 1, 2]


def test_the_builder_starts_a_new_subgroup_after_emitting_one() -> None:
    builder = SubgroupBuilder(tag="X", size=2)
    builder.extend([1.0, 2.0])
    assert builder.pending == ()
    builder.add(3.0)
    assert builder.pending == (3.0,)


def test_pending_shows_what_is_waiting(size: int = 4) -> None:
    builder = SubgroupBuilder(tag="X", size=size)
    builder.add(1.0)
    builder.add(2.0)
    assert builder.pending == (1.0, 2.0)
    assert len(builder) == 2


def test_the_builder_counts_what_it_has_emitted() -> None:
    builder = SubgroupBuilder(tag="X", size=2)
    builder.extend([1.0, 2.0, 3.0, 4.0])
    assert builder.subgroups_emitted == 2


def test_extend_returns_only_completed_subgroups() -> None:
    builder = SubgroupBuilder(tag="X", size=5)
    assert builder.extend([1.0, 2.0, 3.0]) == []
    assert len(builder.extend([4.0, 5.0, 6.0])) == 1


def test_part_indices_and_timestamps_travel_with_the_values() -> None:
    builder = SubgroupBuilder(tag="X", size=2)
    when = datetime(2026, 8, 26, 12, 0, 0)
    builder.add(1.0, part_index=10, timestamp=when)
    group = builder.add(2.0, part_index=11, timestamp=when)
    assert group is not None
    assert group.part_indices == (10, 11)
    assert group.timestamps == (when, when)


def test_a_missing_part_index_is_recorded_as_minus_one() -> None:
    builder = SubgroupBuilder(tag="X", size=2)
    builder.add(1.0)
    group = builder.add(2.0)
    assert group is not None
    assert group.part_indices == (-1, -1)


# --------------------------------------------------------------------------
# Resetting and stoppages
# --------------------------------------------------------------------------


def test_reset_clears_everything(size: int = 2) -> None:
    builder = SubgroupBuilder(tag="X", size=size)
    builder.extend([1.0, 2.0, 3.0])
    builder.reset()
    assert builder.pending == ()
    assert builder.subgroups_emitted == 0


def test_a_stoppage_discards_the_partial_subgroup_but_keeps_the_count() -> None:
    """Parts from either side of a break are not consecutive, so do not mix them."""
    builder = SubgroupBuilder(tag="X", size=5)
    builder.extend([1.0, 2.0, 3.0, 4.0, 5.0])  # one complete subgroup
    builder.extend([100.0, 101.0])  # partial, from before a stoppage
    builder.discard_partial()
    assert builder.pending == ()
    assert builder.subgroups_emitted == 1

    group = builder.extend([1.0, 2.0, 3.0, 4.0, 5.0])[0]
    assert group.values == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert group.index == 1


# --------------------------------------------------------------------------
# Builder validation
# --------------------------------------------------------------------------


def test_a_builder_needs_a_tag_name() -> None:
    with pytest.raises(ValueError, match="tag name"):
        SubgroupBuilder(tag="", size=5)


def test_a_subgroup_size_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        SubgroupBuilder(tag="X", size=1)


def test_a_subgroup_size_with_no_standard_constants_is_rejected() -> None:
    with pytest.raises(ValueError, match="no standard constants"):
        SubgroupBuilder(tag="X", size=26)


def test_the_largest_supported_size_is_accepted() -> None:
    assert SubgroupBuilder(tag="X", size=25).size == 25


# --------------------------------------------------------------------------
# The offline helper
# --------------------------------------------------------------------------


def test_splitting_a_list_gives_complete_subgroups_only() -> None:
    groups = subgroups_from_values([1.0] * 12, size=5)
    assert len(groups) == 2  # 12 values give two full subgroups, two left over


def test_a_short_remainder_is_dropped_rather_than_plotted() -> None:
    """A subgroup of three would have a systematically smaller range."""
    groups = subgroups_from_values(list(range(7)), size=5)
    assert len(groups) == 1
    assert groups[0].values == (0.0, 1.0, 2.0, 3.0, 4.0)


def test_splitting_fewer_values_than_one_subgroup_gives_nothing() -> None:
    assert subgroups_from_values([1.0, 2.0], size=5) == []


def test_the_tag_name_travels_onto_every_subgroup() -> None:
    groups = subgroups_from_values([1.0] * 10, size=5, tag="Torque")
    assert all(g.tag == "Torque" for g in groups)