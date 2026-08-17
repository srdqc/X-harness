"""Plugin runtime context.

A factory (the ``module.path:callable`` named in a manifest) receives
exactly one :class:`PluginContext`. From it, the factory pulls:

- ``config``  — the plugin's own config slice from PicoConfig. The registry
  passes the dict through verbatim; each factory validates the fields it
  consumes. Manifest ``config_schema`` is descriptive metadata today.
- ``services`` — a :class:`ServiceLocator` exposing only the host
  services a backend is allowed to touch. The locator is intentionally
  narrow so plugins don't grow ambient dependencies on arbitrary host
  internals — every field here is a deliberate capability grant.
- ``logger`` — a logger pre-bound with ``plugin=<id>`` so plugin output
  is grep-able in mixed logs.

The locator is a frozen dataclass: factories cannot mutate the host's
view of available services, only read from it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ServiceLocator:
    """Narrow grant of host services to a plugin factory.

    Fields land here as the host needs to expose them. PG-1 starts with
    the bare minimum so the seam exists; later PRs add ``provider`` /
    ``bus`` / etc. as concrete backends prove they need them. We keep
    the dataclass frozen on purpose — every field is a capability, and
    we want adds to be explicit edits to this file, not ambient setattr.
    """

    workspace: Path
    """Root path of the Workspace the plugin operates on."""


@dataclass(frozen=True)
class PluginContext:
    """What a plugin factory sees at activation time."""

    config: dict[str, Any]
    services: ServiceLocator
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("pico.plugin"),
    )


__all__ = ["PluginContext", "ServiceLocator"]
