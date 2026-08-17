"""ConfirmBroker for TUI-RPC confirmation round-trips.

Mirrors :class:`SubscriptionEmitter`: owned by
the RPC server, constructed with ``send_frame`` bound to ``RpcServer.send_frame``,
and passed into ``register_confirm_methods(dispatcher, confirm_broker=...)``.

Transport fail-safe paths (hard-limit timeout, connection EOF via
:meth:`cancel_all`, internal error) resolve to the prompt's default. Task
cancellation propagates.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from loguru import logger

# 单次确认在后端保持挂起的硬上限，与前端可见的 30 秒倒计时无关；
# 35 = 30 + 5 秒网络余量。超时后安全回退到提示框的默认选项。
_CONFIRM_HARD_LIMIT_S = 35.0

SendFrame = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class _PendingConfirm:
    future: asyncio.Future
    default: bool


class ConfirmBroker:
    """Emits ``confirm.request`` notifications and awaits ``confirm.respond``."""

    def __init__(self, send_frame: SendFrame) -> None:
        self._send_frame = send_frame
        self._pending: dict[str, _PendingConfirm] = {}

    async def await_confirm(self, prompt: str, *, default: bool) -> bool:
        """Emit a ``confirm.request`` and await the matching answer.

        Returns ``default`` on hard-limit timeout, EOF (:meth:`cancel_all`), or
        any internal error. External task cancellation propagates.
        """
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = _PendingConfirm(future=future, default=default)
        try:
            await self._send_frame(
                {
                    "jsonrpc": "2.0",
                    "method": "confirm.request",
                    "params": {
                        "request_id": request_id,
                        "prompt": prompt,
                        "default": default,
                    },
                }
            )
            return await asyncio.wait_for(future, _CONFIRM_HARD_LIMIT_S)
        except asyncio.TimeoutError:
            return default
        except Exception:  # noqa: BLE001 — 失效保护：工作线程必须得到布尔值
            logger.exception("confirm_broker: await_confirm failed for {}", request_id)
            return default
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str, answer: bool) -> bool:
        """Resolve a pending confirm. Idempotent: unknown/done → ``False``."""
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return False
        pending.future.set_result(answer)
        return True

    def cancel_all(self) -> None:
        """Fail-safe every pending confirm to its default (connection EOF)."""
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_result(pending.default)


__all__ = ["ConfirmBroker", "SendFrame", "_CONFIRM_HARD_LIMIT_S"]
