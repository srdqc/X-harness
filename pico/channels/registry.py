"""Auto-discovery for channel adapters — no hardcoded registry."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pico.channels.contract import ChannelSpec

_ADAPTERS_PKG = "pico.channels.adapters"


def discover_specs() -> dict[str, ChannelSpec]:
    """Return ``{name: ChannelSpec}`` for migrated adapters, keyed by package
    name.

    Imports only each ``<name>/spec.py`` (cheap — the heavy SDK import is
    deferred into the spec's ``factory``). An adapter without a ``spec.py`` is
    skipped.
    """
    import pico.channels.adapters as pkg

    specs: dict[str, ChannelSpec] = {}
    for _, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if not ispkg:
            continue
        try:
            mod = importlib.import_module(f"{_ADAPTERS_PKG}.{name}.spec")
        except ModuleNotFoundError:
            continue  # 尚未迁移
        if (spec := getattr(mod, "SPEC", None)) is not None:
            specs[name] = spec
    return specs


def discover_channel_names() -> list[str]:
    """Return adapter names by scanning the adapters package (zero imports).

    Enumerates both flat modules and adapter sub-packages. The scan is one
    level deep, so helper modules nested
    inside an adapter sub-package are not listed and never get mistaken
    for a channel.
    """
    import pico.channels.adapters as pkg

    return sorted(name for _, name, _ in pkgutil.iter_modules(pkg.__path__))
