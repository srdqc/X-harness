"""Agent core module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pico.agent.context import ContextBuilder
    from pico.agent.loop import AgentLoop
    from pico.memory_engine.consolidate.consolidator import MemoryStore

__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore"]

# 通过 PEP 562 延迟重新导出：导入 ``pico.agent`` 子模块时不应急切构建 ``AgentLoop``，
# 否则会进一步导入 litellm，成为 CLI 冷启动的主要开销。
_LAZY_EXPORTS = {
    "ContextBuilder": "pico.agent.context",
    "AgentLoop": "pico.agent.loop",
    "MemoryStore": "pico.memory_engine.consolidate.consolidator",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
