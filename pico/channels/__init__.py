"""Chat channels module.

Public contract surface — import channel types from here, not from internal
modules, so the file layout can change without breaking callers.
"""

from pico.channels.base import ChannelBase
from pico.channels.contract import (
    Capabilities,
    Channel,
    ChannelSpec,
    SupportsLogin,
    SupportsStreaming,
)
from pico.channels.manager import ChannelManager

# 公共接口即适配器实现的契约类型。校验辅助函数
# （capability_violations）位于 channels.contract。
__all__ = [
    "Capabilities",
    "Channel",
    "ChannelBase",
    "ChannelManager",
    "ChannelSpec",
    "SupportsLogin",
    "SupportsStreaming",
]
