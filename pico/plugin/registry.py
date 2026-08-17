"""Plugin registry - admits manifests and resolves factories on demand.

Two responsibilities, split deliberately:

1. **Activation** (:meth:`activate`) admits manifests according to config,
   validates contribution conflicts, and records unresolved factory refs. It
   does not import plugin Python or widen ``sys.path``.

2. **Build** resolves and caches only the factory for a backend or Tool that a
   host actually constructs.

Across-manifest name conflicts (two activated plugins both contributing
the same memory backend name) raise :class:`PluginConflictError` -
which the host treats as a startup failure. The discovery layer already
deduplicated *plugins* by id; the registry adds the second layer of
deduplication on *contribution names*.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pico.plugin.discover import DiscoveredPlugin, Source
from pico.plugin.manifest import PluginManifest
from pico.tracing import semconv, trace

logger = logging.getLogger(__name__)


# 记忆后端工厂是接收 PluginContext 并返回 MemoryBackend 实现的可调用对象。
# MemoryBackend 在 MB-1 落地；在此之前返回值标为 Any，使 PG 可以独立编译。
MemoryBackendFactory = Callable[[Any], Any]

# 工具工厂接收 PluginContext 并返回单个工具。
# ``pico.agent.tools.base.Tool``。此处标为 Any，使插件
# 保持本层轻量导入，不依赖 agent 包。
ToolFactory = Callable[[Any], Any]


class PluginError(Exception):
    """Base for plugin-system errors. Catchable as a single class so
    CLI / host startup can render a unified diagnostic banner."""


class PluginConflictError(PluginError):
    """Two activated plugins contributed the same name into one slot."""


class PluginFactoryImportError(PluginError):
    """A manifest pointed at ``module.path:callable`` we couldn't import
    or resolve."""


class PluginNotFoundError(PluginError):
    """The user asked for a backend name no activated plugin contributes."""


@dataclass(frozen=True)
class _ActivatedFactory:
    """Unresolved factory reference plus provenance for diagnostics."""

    plugin_id: str
    name: str
    factory_ref: str
    source: Source
    location: Path | None


class PluginRegistry:
    """Single registration center for activated contribution factories."""

    def __init__(self) -> None:
        self._manifests: dict[str, PluginManifest] = {}
        self._memory_backends: dict[str, _ActivatedFactory] = {}
        self._tools: dict[str, _ActivatedFactory] = {}
        self._resolved_factories: dict[tuple[str, str], MemoryBackendFactory] = {}

    # ── 激活 ─────────────────────────────────────────────────────

    def activate(
        self,
        discovered: list[DiscoveredPlugin],
        *,
        disabled: frozenset[str] = frozenset(),
    ) -> None:
        """Admit manifests without importing Plugin Python code.

        A plugin is admitted iff:

        - its id is not in ``disabled`` (user opt-out), AND
        - ``enabled_by_default`` is True OR the host has another reason
          to include it. PG-2 enforces only the first rule; PG-3 layers
          on the second when wired to the user config.
        """
        for d in discovered:
            mf = d.manifest
            if mf.id in disabled:
                logger.info("plugin %s disabled by user config", mf.id)
                continue
            if not mf.enabled_by_default:
                logger.info(
                    "plugin %s not enabled by default; skipping (use explicit opt-in once supported)",
                    mf.id,
                )
                continue
            self._activate_one(mf, source=d.source, location=d.location)

    def _activate_one(
        self,
        mf: PluginManifest,
        *,
        source: Source,
        location: Path | None,
    ) -> None:
        if mf.id in self._manifests:
            # Discovery 理应已经去重；此处属于防御性检查。
            raise PluginConflictError(
                f"plugin id {mf.id!r} activated twice",
            )
        pending_memory_backends: dict[str, _ActivatedFactory] = {}
        pending_tools: dict[str, _ActivatedFactory] = {}

        for contribution in mf.contributes.memory_backends:
            if contribution.name in self._memory_backends:
                prev = self._memory_backends[contribution.name]
                raise PluginConflictError(
                    f"memory_backend {contribution.name!r} contributed by both {prev.plugin_id!r} and {mf.id!r}",
                )
            pending_memory_backends[contribution.name] = _ActivatedFactory(
                plugin_id=mf.id,
                name=contribution.name,
                factory_ref=contribution.factory,
                source=source,
                location=location,
            )

        for tool in mf.contributes.tools:
            if tool.name in self._tools:
                prev = self._tools[tool.name]
                raise PluginConflictError(
                    f"tool {tool.name!r} contributed by both {prev.plugin_id!r} and {mf.id!r}",
                )
            pending_tools[tool.name] = _ActivatedFactory(
                plugin_id=mf.id,
                name=tool.name,
                factory_ref=tool.factory,
                source=source,
                location=location,
            )

        self._manifests[mf.id] = mf
        self._memory_backends.update(pending_memory_backends)
        self._tools.update(pending_tools)
        for contribution in mf.contributes.memory_backends:
            logger.debug("registered memory_backend %s from %s", contribution.name, mf.id)
        for tool in mf.contributes.tools:
            logger.debug("registered tool %s from %s", tool.name, mf.id)

    @staticmethod
    def _ensure_importable(source: Source, location: Path | None) -> None:
        """Expose a file-based Plugin directory immediately before import.

        USER and explicit PROJECT Plugins may ship code next to the manifest.
        Their directory is appended only when a caller resolves a contributed
        factory. Automatic project discovery is disabled at the host boundary.
        """
        if source not in (Source.USER, Source.PROJECT) or location is None:
            return
        plugin_dir = str(location.parent)
        if plugin_dir not in sys.path:
            sys.path.append(plugin_dir)

    @staticmethod
    def _resolve_factory(plugin_id: str, ref: str) -> MemoryBackendFactory:
        """Import ``module`` and grab ``callable`` from it.

        Manifest validation already enforced the ``module.path:callable``
        shape, so this just splits and imports.
        """
        module_path, attr = ref.split(":", 1)
        try:
            mod = importlib.import_module(module_path)
        except Exception as e:
            raise PluginFactoryImportError(
                f"plugin {plugin_id!r}: importing {module_path!r} failed: {e}",
            ) from e
        try:
            obj = getattr(mod, attr)
        except AttributeError as e:
            raise PluginFactoryImportError(
                f"plugin {plugin_id!r}: {module_path!r} has no attribute {attr!r}",
            ) from e
        if not callable(obj):
            raise PluginFactoryImportError(
                f"plugin {plugin_id!r}: {ref} resolved to a non-callable {type(obj).__name__}",
            )
        return obj  # type: ignore[return-value]

    def _get_factory(
        self,
        kind: str,
        entry: _ActivatedFactory,
    ) -> MemoryBackendFactory:
        key = (kind, entry.name)
        cached = self._resolved_factories.get(key)
        if cached is not None:
            return cached
        self._ensure_importable(entry.source, entry.location)
        factory = self._resolve_factory(entry.plugin_id, entry.factory_ref)
        self._resolved_factories[key] = factory
        return factory

    # ── 内省 ─────────────────────────────────────────────────────

    def activated_ids(self) -> list[str]:
        """Stable-ordered list of activated plugin ids."""
        return sorted(self._manifests)

    def memory_backend_names(self) -> list[str]:
        """Stable-ordered list of registered memory-backend names."""
        return sorted(self._memory_backends)

    def get_memory_backend_factory(self, name: str) -> MemoryBackendFactory:
        """Resolve the backend factory for ``name`` on first use."""
        try:
            entry = self._memory_backends[name]
        except KeyError as e:
            raise PluginNotFoundError(
                f"no memory_backend named {name!r} (registered: {self.memory_backend_names()})",
            ) from e
        return self._get_factory("memory_backend", entry)

    def memory_backend_identity(self, name: str) -> tuple[str, str]:
        """Return the owning plugin id and declared factory reference."""
        try:
            entry = self._memory_backends[name]
        except KeyError as e:
            raise PluginNotFoundError(
                f"no memory_backend named {name!r} (registered: {self.memory_backend_names()})",
            ) from e
        return entry.plugin_id, entry.factory_ref

    def tool_names(self) -> list[str]:
        """Stable-ordered list of registered plugin-tool names."""
        return sorted(self._tools)

    def tool_plugin_id(self, name: str) -> str | None:
        """Plugin id that contributed tool ``name``, or ``None``."""
        entry = self._tools.get(name)
        return entry.plugin_id if entry is not None else None

    def get_tool_factory(self, name: str) -> ToolFactory:
        """Resolve the Tool factory for ``name`` on first use."""
        try:
            entry = self._tools[name]
        except KeyError as e:
            raise PluginNotFoundError(
                f"no tool named {name!r} (registered: {self.tool_names()})",
            ) from e
        return self._get_factory("tool", entry)

    def manifest_for(self, plugin_id: str) -> PluginManifest | None:
        """Return the manifest of an activated plugin, or None."""
        return self._manifests.get(plugin_id)

    # ── 构建（PG-3 入口）─────────────────────────────────────────

    @trace.instrument("plugin.load", extract=semconv.plugin_load("memory_backend"))
    def build_memory_backend(
        self,
        name: str,
        *,
        config: dict[str, Any],
        services: "ServiceLocator",
        logger: logging.Logger | None = None,
    ) -> Any:
        """Resolve the named factory and call it with a fresh ``PluginContext``.

        Construction is synchronous — factories that need async setup
        return a backend whose ``start()`` will be awaited later by the
        host. Any exception from the factory propagates so the host
        sees the real cause rather than a wrapped one.
        """
        from pico.plugin.context import PluginContext  # 局部导入以规避循环依赖

        factory = self.get_memory_backend_factory(name)
        ctx = PluginContext(
            config=config,
            services=services,
            logger=logger or logging.getLogger(f"pico.plugin.{name}"),
        )
        return factory(ctx)

    @trace.instrument("plugin.load", extract=semconv.plugin_load("tool"))
    def build_tool(
        self,
        name: str,
        *,
        config: dict[str, Any],
        services: "ServiceLocator",
        logger: logging.Logger | None = None,
    ) -> Any:
        """Resolve the named tool factory and call it with a fresh
        ``PluginContext``, returning the constructed ``Tool``.

        Symmetric with :meth:`build_memory_backend`: synchronous
        construction, exceptions propagate so the host sees the real
        cause. The host registers the returned tool into the agent's
        :class:`ToolRegistry`.
        """
        from pico.plugin.context import PluginContext  # 局部导入以规避循环依赖

        factory = self.get_tool_factory(name)
        ctx = PluginContext(
            config=config,
            services=services,
            logger=logger or logging.getLogger(f"pico.plugin.{name}"),
        )
        return factory(ctx)


# 为上方类型提示提供前向导入。放在模块末尾，既延后导入成本，
# 也避免模块加载时触发循环依赖（registry 会被
# 在上下文就绪前调用 __init__）。
from pico.plugin.context import ServiceLocator  # noqa: E402

__all__ = [
    "MemoryBackendFactory",
    "PluginConflictError",
    "PluginError",
    "PluginFactoryImportError",
    "PluginNotFoundError",
    "PluginRegistry",
    "ToolFactory",
]
