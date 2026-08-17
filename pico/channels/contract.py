"""The channel contract — what a chat-channel adapter must satisfy and declare.

A channel implements the :class:`Channel` protocol (``start``/``stop``/``send``)
and declares its :class:`Capabilities`; optional behaviours are separate
``Supports*`` protocols a channel opts into. Each channel package exports a
:class:`ChannelSpec` — a lightweight descriptor whose ``factory`` defers the
heavy SDK import — consumed by the registry.

Composition over inheritance: there is no base class to subclass. Adapters
satisfy the protocols structurally and inject the framework services
(:mod:`.intake`, transcription) they need.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

# Capabilities 和 SupportsStreaming 定义在 spine.delivery（其消费者是
# delivery hub）；此处重新导出，让 channel 始终从同一入口导入。
from pico.spine.delivery import Capabilities, SupportsStreaming

if TYPE_CHECKING:
    from pico.channels.intake import Intake


@runtime_checkable
class Channel(Protocol):
    """Minimal required contract every channel satisfies."""

    name: str
    capabilities: Capabilities
    intake: Intake

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, chat_id: str, content: str, media: list[str] | None = None) -> None: ...


@runtime_checkable
class SupportsLogin(Protocol):
    """Opt-in interactive (QR/scan) login, run once via CLI before ``start``."""

    async def login(self, force: bool = False) -> bool: ...


Maturity = Literal["beta", "live-gated"]


@dataclass(frozen=True)
class ChannelSpec:
    """Declarative descriptor a channel package exports as ``SPEC``.

    ``factory`` defers the channel's heavy SDK import, so collecting specs
    (listing / onboarding / login routing) stays cheap. Carries only what can't
    be located elsewhere: the channel's name is its package name (the registry
    key); dependency/setup guidance is derived by the CLI from capabilities +
    the config schema.

    ``maturity`` names the evidence level behind the adapter, not code quality:
    ``beta`` = deterministic contract (V-C0) and security (V-S0) bundles only;
    ``live-gated`` = a live Channel gate has also passed. It is the single
    source the CLI, doctor, and onboarding read, so no surface can claim a
    higher level than the spec declares.
    """

    display_name: str
    factory: Callable[[Any], Channel]  # 配置映射到渠道实例
    capabilities: Capabilities = field(default_factory=Capabilities)
    maturity: Maturity = "beta"


# 每个 capability 标志必须与对应的 opt-in protocol 一致。新增 capability
# 时增加一行即可；下方检查会同时覆盖两个方向。
_CAP_PROTOCOLS: tuple[tuple[str, type], ...] = (
    ("interactive_login", SupportsLogin),
    ("streaming", SupportsStreaming),
)


def capability_violations(channel: object, caps: Capabilities | None = None) -> list[str]:
    """Return mismatches between declared capabilities and implemented protocols.

    A channel declaring a capability must implement the matching ``Supports*``
    protocol, and vice-versa. Empty list = consistent. Used by the per-channel
    capability-proof tests.
    """
    caps = caps if caps is not None else getattr(channel, "capabilities", Capabilities())
    out: list[str] = []
    for flag, proto in _CAP_PROTOCOLS:
        declared = getattr(caps, flag)
        implemented = isinstance(channel, proto)
        if declared and not implemented:
            out.append(f"declares {flag} but does not implement {proto.__name__}")
        elif implemented and not declared:
            out.append(f"implements {proto.__name__} but does not declare {flag}")
    return out
