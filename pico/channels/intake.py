"""Inbound intake — the framework service that turns a channel's raw inbound
into a permission-checked spine ``TurnRequest`` submitted to the scheduler.

Injected into adapters (composition, not inheritance): deny-by-policy, then
submit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from pico.auth.allowlist import is_allowed
from pico.spine import ChatType, Media, Origin, Source, TurnRequest


class Intake:
    """Per-channel inbound gate + submitter. One instance per channel.

    ``allow_check`` overrides the default allowlist policy for channels with
    bespoke permission rules when a platform needs them.

    A permitted message is handed to the spine dispatch (wired via
    ``set_submit``) as a ``TurnRequest`` — channel metadata rides
    ``Source.extras``.
    """

    def __init__(
        self,
        channel_name: str,
        config: Any,
        allow_check: Callable[[str], bool] | None = None,
    ):
        self.channel_name = channel_name
        self.config = config
        self._allow_check = allow_check
        self._submit: Callable[[TurnRequest], Awaitable[None]] | None = None
        self._sealed = False
        self._inflight = 0
        self._inflight_tasks: set[asyncio.Task[Any]] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    def set_submit(self, submit: Callable[[TurnRequest], Awaitable[None]]) -> None:
        """Wire the spine inbound dispatch (gateway). ``publish`` submits a
        TurnRequest through it. The dispatch is control-aware (it intercepts
        /stop and /restart) — Intake stays a dumb gate + builder; the
        control/cancel logic lives where the scheduler and agent are (the
        gateway)."""
        self._submit = submit

    def is_allowed(self, sender_id: str) -> bool:
        """Deny-by-default allowlist check (``*`` = all; empty = deny all),
        unless a custom ``allow_check`` was supplied."""
        if self._allow_check is not None:
            return self._allow_check(sender_id)
        return is_allowed(self.channel_name, sender_id, getattr(self.config, "allow_from", None))

    def seal(self) -> None:
        """Permanently reject new inbound publishes during gateway shutdown."""
        self._sealed = True

    async def wait_idle(self) -> None:
        """Wait until every publish admitted before ``seal`` has returned."""
        await self._idle.wait()

    def cancel_inflight(self) -> int:
        """Cancel publishes admitted before shutdown and return their count."""
        tasks = tuple(self._inflight_tasks)
        for task in tasks:
            task.cancel()
        return len(tasks)

    async def publish(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """Check permission, then submit the message to the spine dispatch as a
        TurnRequest."""
        if self._sealed:
            logger.info(
                "Intake for channel {} is sealed; dropping inbound from {}",
                self.channel_name,
                sender_id,
            )
            return
        if not self.is_allowed(sender_id):
            logger.warning(
                "Access denied for sender {} on channel {}. Add them to allowFrom list in config to grant access.",
                sender_id,
                self.channel_name,
            )
            return

        if self._submit is None:
            logger.error(
                "Intake for channel {} has no spine dispatch wired; dropping inbound from {}",
                self.channel_name,
                sender_id,
            )
            return

        self._inflight += 1
        task = asyncio.current_task()
        if task is not None:
            self._inflight_tasks.add(task)
        self._idle.clear()
        try:
            meta = metadata or {}
            await self._submit(
                TurnRequest(
                    origin=Origin.USER,
                    source=Source(
                        channel=self.channel_name,
                        chat_id=str(chat_id),
                        sender_id=str(sender_id),
                        # 该值不承载处理语义（channel 从 extras 携带的 metadata
                        # 读取真实 chat_type），这里只为 Spine 尽力提供形状：
                        # channel 声明为 group 时使用 group，否则使用 DM。
                        chat_type=ChatType.GROUP if meta.get("chat_type") == "group" else ChatType.DM,
                        extras=meta,
                    ),
                    text=content,
                    media=tuple(Media(path=p, mime="application/octet-stream", kind="file") for p in (media or [])),
                    # session_key_override -> conversation：run_turn 的 cid
                    # 使用 `conversation or channel:chat_id`。
                    conversation=session_key,
                )
            )
        finally:
            if task is not None:
                self._inflight_tasks.discard(task)
            self._inflight -= 1
            if self._inflight == 0:
                self._idle.set()
