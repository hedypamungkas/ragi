"""ragworkbench/log -- stdlib logging helper.

Replaces koboi's ``AgentLogger`` with a thin ``logging.getLogger`` wrapper so the
library has no custom logger dependency. Callers can attach handlers/config as usual.
"""

from __future__ import annotations

import logging

_LOGGER_NAME = "ragworkbench"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a logger under the ragworkbench namespace."""
    return logging.getLogger(name)
