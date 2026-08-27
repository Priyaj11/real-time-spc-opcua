"""Fixtures shared by every test module.

pytest loads this file automatically before any test runs, and everything
defined here is available to every test in the folder without an import. That
makes it the right home for the two things the suite kept repeating: loading
machine.yaml, and turning a fault into subgroups.

The simulation itself is not written here. It lives in
spc_opcua.simulator.offline, because Milestone 12's scenario evaluation needs
exactly the same thing and a study script should not have to import from a test
folder. What this file adds is fixture packaging.

Fixture scope
-------------
These fixtures are session-scoped, so they are built once for the whole run
rather than once per test. That is only safe because everything they hand out
is read-only: MachineConfig is a frozen dataclass, Subgroup is frozen, and no
test mutates the lists it receives. A fixture holding anything a test could
modify must stay function-scoped, or one test will quietly corrupt the next.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from spc_opcua.config import MachineConfig, TagSpec, load_config
from spc_opcua.simulator.faults import Fault, ToolWear
from spc_opcua.simulator.offline import DEFAULT_TAG, bore_subgroups
from spc_opcua.spc.subgroups import Subgroup

# Modules whose tests open a real socket. Kept as a tuple rather than a
# decorator on each test, because a mislabelled integration test is how a fast
# loop silently becomes a slow one.
INTEGRATION_MODULES = ("test_opcua_server", "test_opcua_client", "test_live_source")

SubgroupFactory = Callable[..., list[Subgroup]]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test in an integration module, so nobody has to remember to."""
    for item in items:
        if any(name in item.nodeid for name in INTEGRATION_MODULES):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def config() -> MachineConfig:
    """The machine definition from config/machine.yaml, parsed once per run."""
    return load_config()


@pytest.fixture(scope="session")
def bore_spec(config: MachineConfig) -> TagSpec:
    """The specification for the characteristic every SPC test charts."""
    return config.tag(DEFAULT_TAG)


@pytest.fixture(scope="session")
def make_subgroups(config: MachineConfig) -> SubgroupFactory:
    """A factory that turns a fault list into bore subgroups.

    Returned as a callable rather than as data, because every test wants a
    different fault, seed and length. This is the factory-as-fixture pattern:
    the fixture assembles the machinery once, and each test calls it with its
    own arguments.
    """

    def build(
        faults: Sequence[Fault] = (), seed: int = 1, count: int = 25
    ) -> list[Subgroup]:
        return bore_subgroups(faults, seed=seed, count=count, config=config)

    return build


@pytest.fixture(scope="session")
def healthy_subgroups(make_subgroups: SubgroupFactory) -> SubgroupFactory:
    """Subgroups from a machine with nothing wrong with it."""

    def build(seed: int = 1, count: int = 25) -> list[Subgroup]:
        return make_subgroups(seed=seed, count=count)

    return build


@pytest.fixture(scope="session")
def worn_subgroups(make_subgroups: SubgroupFactory) -> SubgroupFactory:
    """Subgroups from a machine whose boring tool is wearing down."""

    def build(seed: int = 1, count: int = 40, rate: float = -0.05) -> list[Subgroup]:
        return make_subgroups(
            [ToolWear(tag=DEFAULT_TAG, rate_per_hour=rate)], seed=seed, count=count
        )

    return build