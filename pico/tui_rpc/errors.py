"""Custom RPC exception classes mapped to JSON-RPC 2.0 error codes.

Code table — frozen in `specs/tui-ipc.md` §4 (server-defined range -32000..-32099):

| code   | message                       | meaning                          |
|--------|-------------------------------|----------------------------------|
| -32001 | session_not_found             | session_key unknown              |
| -32002 | session_locked                | session held by another client   |
| -32003 | turn_in_progress              | session currently running a turn |
| -32008 | model_not_available           | model_id not routable            |
| -32009 | model_switch_in_turn          | switch attempt while turn live   |
| -32010 | config_field_readonly         | not on hot-changeable whitelist  |
| -32011 | config_validation_error       | Pydantic / semver validation     |
| -32012 | not_supported_in_v01          | unsupported provider auth action |

JSON-RPC pre-defined codes (-32700/-32600/-32601/-32602) are emitted directly
by the dispatcher and have no dedicated exception class.

-32603 ``internal_error`` is also dispatcher-emitted for uncaught handler
exceptions, but it has a dedicated ``InternalError`` class so non-dispatcher
code-paths (notably the ``_build_tui_agent_loop`` factory invoked from
``_spawn_agent_loop_task``) can raise typed -32603 cross-module. Without it
those paths conflated init crashes into -32008 ``model_not_available``.
"""

from __future__ import annotations

from typing import Any, ClassVar


class RpcError(Exception):
    """Base class for all RPC errors mapped to JSON-RPC error frames.

    Subclasses set the class-level `CODE` and `MESSAGE` constants; the
    dispatcher reads them when serializing the error frame. `data` is
    optional structured context (echoed into JSON-RPC `error.data`).
    """

    CODE: ClassVar[int] = -32099  # 兜底错误
    MESSAGE: ClassVar[str] = "rpc_error"

    def __init__(self, detail: str = "", data: dict[str, Any] | None = None):
        self.detail = detail
        self.data = data
        # 默认 str() 展示错误码、消息和可选详情
        super().__init__(f"{self.MESSAGE}: {detail}" if detail else self.MESSAGE)

    @property
    def code(self) -> int:
        return self.CODE

    @property
    def message(self) -> str:
        return self.MESSAGE

    @property
    def public_message(self) -> str:
        if isinstance(self.data, dict):
            value = self.data.get("public_message")
            if isinstance(value, str) and value.strip():
                return value
        return self.message


class SessionNotFoundError(RpcError):
    CODE = -32001
    MESSAGE = "session_not_found"


class SessionLockedError(RpcError):
    CODE = -32002
    MESSAGE = "session_locked"


class TurnInProgressError(RpcError):
    CODE = -32003
    MESSAGE = "turn_in_progress"


class ModelNotAvailableError(RpcError):
    CODE = -32008
    MESSAGE = "model_not_available"


class ModelSwitchInTurnError(RpcError):
    CODE = -32009
    MESSAGE = "model_switch_in_turn"


class ConfigFieldReadonlyError(RpcError):
    CODE = -32010
    MESSAGE = "config_field_readonly"


class ConfigValidationError(RpcError):
    CODE = -32011
    MESSAGE = "config_validation_error"


class NotSupportedInV01Error(RpcError):
    CODE = -32012
    MESSAGE = "not_supported_in_v01"


# 后续扩展区间（-32016..-32049）。-32016 表示订阅溢出；早期草案曾误用 -32010，
# 但 -32010 已属于 ConfigFieldReadonlyError。
class SubscriptionCapacityExceededError(RpcError):
    CODE = -32016
    MESSAGE = "subscription_capacity_exceeded"


# JSON-RPC 预定义的 ``internal_error``（-32603）。添加此类后，非分派器路径也能
# 跨模块抛出带类型的 -32603。例如 ``_build_tui_agent_loop`` 运行在处理器上下文之外，
# 却需通过 ``_spawn_agent_loop_task`` 使用的工厂闭包暴露初始化崩溃。
class InternalError(RpcError):
    CODE = -32603
    MESSAGE = "internal_error"


# JSON-RPC 预定义错误码（见 specs §2.3 / RFC）。
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# 反向查找：JSON-RPC 错误码 → 异常类，用于客户端重建异常或测试断言。
JSONRPC_ERROR_REGISTRY: dict[int, type[RpcError]] = {
    cls.CODE: cls
    for cls in (
        SessionNotFoundError,
        SessionLockedError,
        TurnInProgressError,
        ModelNotAvailableError,
        ModelSwitchInTurnError,
        ConfigFieldReadonlyError,
        ConfigValidationError,
        NotSupportedInV01Error,
        SubscriptionCapacityExceededError,
        InternalError,
    )
}


__all__ = [
    "RpcError",
    "SessionNotFoundError",
    "SessionLockedError",
    "TurnInProgressError",
    "ModelNotAvailableError",
    "ModelSwitchInTurnError",
    "ConfigFieldReadonlyError",
    "ConfigValidationError",
    "NotSupportedInV01Error",
    "SubscriptionCapacityExceededError",
    "InternalError",
    "JSONRPC_ERROR_REGISTRY",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
]
