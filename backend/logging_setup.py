"""One place that configures how this app's own log lines look and how verbose they
are, so every module just does `logger = logging.getLogger(__name__)` and gets
consistent, readable output in whichever terminal actually runs the server.

Distinct on purpose from uvicorn's own access-log lines (the "INFO: 127.0.0.1 - GET
... 200 OK" lines) -- those always print regardless of this, and this module's
"aura." prefix on every line makes this app's own pipeline-stage logs easy to
visually pick out from that access-log noise while watching the terminal.
"""

import logging

from .config import settings

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"))

    root = logging.getLogger("aura")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"aura.{name}")
