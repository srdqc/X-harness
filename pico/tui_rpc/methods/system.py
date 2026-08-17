"""`system.*` RPC handlers — handshake, ping, version.

These handlers are invoked by the dispatcher with a plain `params: dict` and
must return a plain `result: dict`. Validation uses Pydantic v2 models from
`pico/tui_rpc/models.py` when available; otherwise we inline a lightweight
semver guard so the dispatcher can be tested standalone.
"""

from __future__ import annotations

import importlib.metadata as _md
import os
import re
import time
from typing import TYPE_CHECKING

from loguru import logger

from pico.tui_rpc.errors import ConfigValidationError

if TYPE_CHECKING:
    from pico.tui_rpc.dispatcher import Dispatcher


# ----------------------------------------------------------------------------
# 版本定义
# ----------------------------------------------------------------------------
# server_version：IPC 桥接协议的实现版本，发布新的线上协议兼容版本时递增。
# schema_version：与 `ui-tui/rpc-schema/openrpc.json` 中的 OpenRPC `info.version` 一致。
# pico_version：从已安装元数据读取的 Pico 发行版本。
SERVER_VERSION = "0.1.0"
SCHEMA_VERSION = "0.1.0"
SERVER_CAPABILITIES = ["jsonrpc-2.0", "subscriptions", "attachments", "sessions", "confirm"]

# 宽松的语义化版本：<major>.<minor>.<patch>，可选 `-prerelease` 或 `+build`。
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def _pico_version() -> str:
    try:
        return _md.version("pico-harness")
    except _md.PackageNotFoundError:
        # CI 中的可编辑安装可能未注册元数据；回退到约定的哨兵值，避免握手崩溃。
        return "0.0.0+unknown"


# ----------------------------------------------------------------------------
# 处理器
# ----------------------------------------------------------------------------


async def system_hello(params: dict) -> dict:
    """`system.hello` — initial handshake. Validates client_version semver.

    Spec: §3.7 `system.hello` — errors -32011 if client_version invalid.
    """
    client_version = params.get("client_version")
    if not isinstance(client_version, str) or not client_version:
        raise ConfigValidationError(
            "client_version is required",
            data={"field": "client_version", "reason": "missing"},
        )
    if not _SEMVER_RE.match(client_version):
        raise ConfigValidationError(
            f"client_version '{client_version}' is not a valid semver",
            data={"field": "client_version", "value": client_version},
        )

    client_capabilities = params.get("client_capabilities", []) or []
    # pid 用于区分共享同一日志文件的并发 `pico` 进程。
    logger.info(
        "tui_rpc: handshake — pid={} client_version={} client_capabilities={}",
        os.getpid(),
        client_version,
        client_capabilities,
    )
    return {
        "server_version": SERVER_VERSION,
        "server_capabilities": list(SERVER_CAPABILITIES),
        "session": {
            "default_channel": "tui",
            "default_session_key": "tui:default",
        },
    }


async def system_ping(params: dict) -> dict:
    """`system.ping` — RTT probe. Returns server timestamp in ms."""
    return {
        "pong": True,
        "server_time_ms": int(time.time() * 1000),
    }


async def system_version(params: dict) -> dict:
    """`system.version` — version triple for diagnostics / compatibility checks."""
    return {
        "server_version": SERVER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "pico_version": _pico_version(),
    }


def register_system_methods(dispatcher: "Dispatcher") -> None:
    """Register all 3 system.* methods on a dispatcher instance."""
    dispatcher.register("system.hello", system_hello)
    dispatcher.register("system.ping", system_ping)
    dispatcher.register("system.version", system_version)


__all__ = [
    "system_hello",
    "system_ping",
    "system_version",
    "register_system_methods",
    "SERVER_VERSION",
    "SCHEMA_VERSION",
    "SERVER_CAPABILITIES",
    "_pico_version",
]
