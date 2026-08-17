"""Memory contracts, host-owned context carriers, and Local Skill routing."""

from typing import TYPE_CHECKING

from pico.memory_engine.backend import Memory, MemoryBackend
from pico.memory_engine.base import AssembledContext, TokenBudget

if TYPE_CHECKING:
    from pico.memory_engine.contract_test import (
        LifecycleContractTests,
        MemoryBackendContractTests,
    )

__all__ = [
    "AssembledContext",
    "LifecycleContractTests",
    "Memory",
    "MemoryBackend",
    "MemoryBackendContractTests",
    "TokenBudget",
]


# 合约测试基类位于 ``contract_test``，该模块会在顶层导入仅开发环境依赖的 ``pytest``。
# 如果在此处急切导入，每次 ``import pico.memory_engine`` 都会引入 pytest，导致未安装
# pytest 的生产环境（如打包后的 `pico`）以 ``ModuleNotFoundError`` 失败。通过 PEP 562
# 延迟暴露，只在测试套件真正访问这些类时解析，此时 pytest 已可用。
def __getattr__(name: str):
    if name in ("LifecycleContractTests", "MemoryBackendContractTests"):
        from pico.memory_engine import contract_test

        return getattr(contract_test, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
