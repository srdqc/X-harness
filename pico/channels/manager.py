"""Channel manager for coordinating chat channels.

Construction + lifecycle only. Outbound delivery is the spine's
DeliveryHub/Outlet (a ChannelOutletAdapter per channel registered by the
gateway); inbound is each channel's Intake -> scheduler.submit.
"""

from __future__ import annotations

import asyncio
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any

from loguru import logger

from pico.channels.contract import Channel
from pico.config.schema import Config
from pico.product import DISTRIBUTION_NAME
from pico.spine._barrier import finish_barrier

_INTAKE_DRAIN_TIMEOUT_S = 5.0
_CHANNEL_STOP_TIMEOUT_S = 5.0


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    task.exception()


def _missing_dep_hint(modname: str) -> str:
    """How to install a channel's missing SDK, tailored to the install mode.

    An editable (dev) checkout uses ``uv sync``; a wheel/tool install has no
    source tree, so it must re-run the installer instead. PEP 610
    ``direct_url.json`` distinguishes them -- a wheel install records
    ``archive_info`` (no ``dir_info`` key), so ``.get`` chaining avoids a
    KeyError when it is absent. This runs while a channel is already failing,
    so a missing/corrupt file must degrade to the installer hint, never raise.
    """
    editable = False
    try:
        raw = distribution(DISTRIBUTION_NAME).read_text("direct_url.json")
        if raw:
            editable = bool(json.loads(raw).get("dir_info", {}).get("editable", False))
    except (PackageNotFoundError, ValueError):
        pass

    if editable:
        return f"Run: uv sync --extra channel-{modname}"
    if sys.platform == "win32":
        return (
            "Re-run the Pico installer to add channels: "
            "$h = @{Authorization=\"Bearer $env:PICO_GITEE_TOKEN\"}; "
            "irm https://gitee.com/htxoffical/pico-harness/raw/main/install.ps1 -Headers $h | iex"
        )
    return (
        "Re-run the Pico installer to add channels: "
        "printf 'Authorization: Bearer %s\\n' \"$PICO_GITEE_TOKEN\" | "
        "curl -fsSL -H @- https://gitee.com/htxoffical/pico-harness/raw/main/install.sh | sh"
    )


class ChannelManager:
    """Manages chat channels: construct enabled adapters, start/stop, status."""

    def __init__(self, config: Config):
        self.config = config
        self.channels: dict[str, Channel] = {}

        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize enabled channels from their declarative ``ChannelSpec``.

        One channel's failure never propagates: a missing SDK, a crashing
        factory, and a deny-all allowlist each disable only that channel, so the
        remaining channels and the Gateway still start.
        """
        from pico.channels.registry import discover_specs

        groq_key = self.config.providers.groq.api_key

        for modname, spec in discover_specs().items():
            section = getattr(self.config.channels, modname, None)
            if not section or not getattr(section, "enabled", False):
                continue
            try:
                channel = spec.factory(section)
                channel.transcription_api_key = groq_key
                self.channels[modname] = channel
                logger.info("{} channel enabled", spec.display_name)
            except ImportError as e:
                logger.warning(
                    "{} channel disabled: missing dependency ({}). {}",
                    modname,
                    e,
                    _missing_dep_hint(modname),
                )
            except Exception as e:
                logger.error(
                    "{} channel disabled: adapter failed to construct ({}: {})",
                    modname,
                    type(e).__name__,
                    e,
                )

        self._validate_allow_from()

    def _validate_allow_from(self) -> None:
        """Disable any channel whose allowlist denies everyone.

        An empty ``allow_from`` is a misconfiguration that would silently drop
        every inbound message, so the channel is dropped loudly instead of being
        left running as a black hole -- and instead of aborting the process,
        which would let one channel's config error take down every other one.
        """
        for name in [n for n, ch in self.channels.items() if getattr(ch.config, "allow_from", None) == []]:
            logger.error(
                '{} channel disabled: empty allowFrom denies every sender. Set ["*"] to allow everyone, '
                "or add specific user IDs.",
                name,
            )
            del self.channels[name]

    async def _start_channel(self, name: str, channel: Channel) -> None:
        """Start a channel and log any exceptions."""
        try:
            await channel.start()
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)

    async def start_all(self) -> None:
        """Start all channels (they run forever). Outbound delivery is the
        spine outlets', not this manager's."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def quiesce_intake(self) -> None:
        """Seal every Intake and bound the drain of admitted publishes."""
        intakes = [channel.intake for channel in self.channels.values()]
        for intake in intakes:
            intake.seal()

        async def wait_idle() -> None:
            await asyncio.gather(*(intake.wait_idle() for intake in intakes))

        # 调用方取消时不得同时取消屏障等待者：先取消已接纳的 publish，
        # 并确认系统进入 idle，之后才向上传播取消。
        drain = asyncio.create_task(wait_idle())
        try:
            await asyncio.wait_for(asyncio.shield(drain), timeout=_INTAKE_DRAIN_TIMEOUT_S)
        except TimeoutError as exc:
            cancelled = sum(intake.cancel_inflight() for intake in intakes)
            logger.error(
                "channel intake drain timed out after {}s; cancelling {} admitted publish(es)",
                _INTAKE_DRAIN_TIMEOUT_S,
                cancelled,
            )
            await finish_barrier(drain)
            raise TimeoutError(
                f"channel intake drain timed out after {_INTAKE_DRAIN_TIMEOUT_S}s; "
                f"cancelled {cancelled} admitted publish(es)"
            ) from exc
        except asyncio.CancelledError as cancellation:
            cancelled = sum(intake.cancel_inflight() for intake in intakes)
            logger.warning(
                "channel intake drain cancelled; cancelling {} admitted publish(es) before spine teardown",
                cancelled,
            )
            await finish_barrier(drain, cancellation=cancellation)
            raise cancellation

    async def _stop_channel(self, name: str, channel: Channel) -> None:
        # 若内部协程吞掉取消，wait_for 可能超过自身超时；独立任务用于严格
        # 执行该截止时间。
        task = asyncio.create_task(channel.stop())
        try:
            done, _pending = await asyncio.wait({task}, timeout=_CHANNEL_STOP_TIMEOUT_S)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(_consume_task_result)
            raise
        if not done:
            task.cancel()
            task.add_done_callback(_consume_task_result)
            raise TimeoutError(f"channel {name} stop timed out after {_CHANNEL_STOP_TIMEOUT_S}s")
        await task

    async def stop_all(self) -> None:
        """Bound and attempt every channel transport stop."""
        logger.info("Stopping all channels...")
        first_error: BaseException | None = None
        cancellation: asyncio.CancelledError | None = None
        for name, channel in self.channels.items():
            try:
                await self._stop_channel(name, channel)
                logger.info("Stopped {} channel", name)
            except asyncio.CancelledError as exc:
                logger.opt(exception=exc).error("Error stopping {}", name)
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                logger.opt(exception=exc).error("Error stopping {}", name)
                if first_error is None:
                    first_error = exc
        if cancellation is not None:
            raise cancellation
        if first_error is not None:
            raise first_error

    def get_channel(self, name: str) -> Channel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {name: {"enabled": True, "running": channel.is_running} for name, channel in self.channels.items()}

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels)
