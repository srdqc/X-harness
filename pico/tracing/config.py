"""Configuration for Pico's in-tree tracing.

Kept light and side-effect-free: read at process startup (from the CLI
``main()`` callback) to decide whether to install instrumentation. Environment
variables are explicit overrides; otherwise the ``[tracing]`` section of the
Pico config file drives behavior, defaulting to on.
"""

from __future__ import annotations

import os
from pathlib import Path

from pico.product import get_product_home

_OFF = {"0", "false", "off", "no"}


def _config_section() -> dict:
    """Read the ``[tracing]`` block from the Pico config file (best-effort).

    Uses Pico's own config-path resolver so a ``--config`` override is honored
    once set. Never raises — tracing must not break startup.
    """
    try:
        import json

        from pico.config.loader import get_config_path

        path = get_config_path()
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
        section = data.get("tracing")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def enabled() -> bool:
    """On by default. ``PICO_TRACING`` env wins; else ``[tracing].enabled``."""
    env = os.environ.get("PICO_TRACING")
    if env is not None:
        return env.strip().lower() not in _OFF
    return bool(_config_section().get("enabled", True))


def state_dir() -> Path:
    """Trace state dir (``~/.pico/traces``). Spans land at ``<dir>/logs/audit-spans.log``.

    Overridable with ``PICO_TRACING_DIR`` (absolute) or ``PICO_HOME``.
    """
    override = os.environ.get("PICO_TRACING_DIR")
    if override:
        return Path(override).expanduser()
    return get_product_home() / "traces"


def port() -> int:
    """Dashboard viewer port. ``TRACING_UI_PORT`` env wins; else ``[tracing].port``."""
    env = os.environ.get("TRACING_UI_PORT")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            return 4318
    try:
        return int(_config_section().get("port", 4318))
    except (ValueError, TypeError):
        return 4318


def preview_len() -> int:
    """Max chars kept inline on a span; full payloads go to artifacts."""
    env = os.environ.get("PICO_TRACING_PREVIEW")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            return 500
    try:
        return max(0, int(_config_section().get("previewLen", 500)))
    except (ValueError, TypeError):
        return 500
