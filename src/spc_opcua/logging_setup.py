"""One place that decides how log messages look for the whole project.

Every module calls logging.getLogger(__name__) and simply logs. Only the
program entry points call configure_logging(), so a library module never
changes logging behaviour behind someone's back.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"
DATE_FORMAT = "%H:%M:%S"

_configured = False


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> None:
    """Set up console logging for the whole application.

    Args:
        level: Minimum level to show, for example logging.INFO or "DEBUG".
        force: Reconfigure even if logging was already configured.
    """
    global _configured
    if _configured and not force:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # asyncua is very chatty at DEBUG level; keep it quieter than our own code.
    logging.getLogger("asyncua").setLevel(logging.WARNING)

    _configured = True
    