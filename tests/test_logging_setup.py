"""Tests for the project's logging configuration.

This module had zero test coverage until Milestone 11's coverage report pointed
at it. It is small, but it is also the reason every other module can call
logging.getLogger(__name__) and simply log, so it is worth pinning down.

Every test here restores the root logger afterwards. Logging is global state,
and a test that leaves a handler behind will change the output, and possibly
the behaviour, of every test that runs after it.
"""

from __future__ import annotations

import logging

import pytest

from spc_opcua import logging_setup
from spc_opcua.logging_setup import DATE_FORMAT, LOG_FORMAT, configure_logging


@pytest.fixture(autouse=True)
def restore_logging():
    """Put the root logger back exactly as it was, whatever the test did."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_flag = logging_setup._configured
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    logging_setup._configured = saved_flag


def test_configuring_installs_exactly_one_handler() -> None:
    logging_setup._configured = False
    configure_logging()
    assert len(logging.getLogger().handlers) == 1


def test_the_level_is_applied_to_the_root_logger() -> None:
    logging_setup._configured = False
    configure_logging(level=logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG


def test_the_level_can_be_given_as_a_word() -> None:
    """Config files and command lines carry strings, not level constants."""
    logging_setup._configured = False
    configure_logging(level="WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_calling_it_twice_does_not_stack_handlers() -> None:
    """Two entry points in one process must not double every log line."""
    logging_setup._configured = False
    configure_logging()
    configure_logging(level=logging.DEBUG)
    assert len(logging.getLogger().handlers) == 1
    # The second call was ignored, so the first call's level still stands.
    assert logging.getLogger().level == logging.INFO


def test_force_lets_an_entry_point_reconfigure_deliberately() -> None:
    logging_setup._configured = False
    configure_logging(level=logging.INFO)
    configure_logging(level=logging.ERROR, force=True)
    assert logging.getLogger().level == logging.ERROR
    assert len(logging.getLogger().handlers) == 1


def test_the_noisy_library_is_kept_quieter_than_our_own_code() -> None:
    """Asyncua logs every packet at DEBUG, which drowns out everything else."""
    logging_setup._configured = False
    configure_logging(level=logging.DEBUG)
    assert logging.getLogger("asyncua").level == logging.WARNING


def test_records_are_formatted_the_way_the_format_string_says(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = logging.LogRecord(
        name="spc_opcua.demo",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="bore drifted",
        args=(),
        exc_info=None,
    )
    formatted = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT).format(record)
    assert "INFO" in formatted
    assert "spc_opcua.demo" in formatted
    assert "bore drifted" in formatted