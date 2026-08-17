"""Pico in-tree tracing: audit.span.v1 observability.

Instrumentation is done by annotating Pico's own methods with the
``@trace.instrument(...)`` decorator (see :mod:`pico.tracing.trace`). Nothing is monkeypatched — the
decorators live in Pico's source and are no-op when tracing is disabled, so a
tracing failure can never alter the host's behavior.

Turn off with ``PICO_TRACING=0`` or ``[tracing] enabled = false`` in the Pico
config. Spans land at ``~/.pico/traces/logs/audit-spans.log`` (override with
``PICO_TRACING_DIR``). Open the dashboard with ``pico tracing`` or ``/tracing``.
"""

from __future__ import annotations

from . import config, trace

__all__ = ["enabled", "trace"]


def enabled() -> bool:
    return config.enabled()
