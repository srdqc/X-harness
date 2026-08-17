"""Spawn tool for creating background subagents."""

from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pico.agent.tools.base import Tool

if TYPE_CHECKING:
    from pico.agent.subagent import SubagentManager


@dataclass(frozen=True)
class _SpawnOrigin:
    """Per-turn origin for subagent announcements, isolated per asyncio task
    (the tool is shared; a turn runs in its own lane task). Frozen +
    copy-on-write so a child task that inherited the parent's value never
    writes back through the shared reference."""

    channel: str
    chat_id: str
    session_key: str


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    # 子 Agent 运行自己的循环，最多 15 次迭代，内部没有墙上时间上限，
    # 因此使用宽裕的兜底超时，而非默认值。
    timeout_seconds = 900.0

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._default = _SpawnOrigin(channel="cli", chat_id="direct", session_key="cli:direct")
        self._origin: ContextVar[_SpawnOrigin] = ContextVar("spawn_origin")

    def _cur(self) -> _SpawnOrigin:
        return self._origin.get(None) or self._default

    def set_context(self, channel: str, chat_id: str, session_key: str) -> None:
        """Set the origin context for subagent announcements (turn-local)."""
        self._origin.set(replace(self._cur(), channel=channel, chat_id=chat_id, session_key=session_key))

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        org = self._cur()
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=org.channel,
            origin_chat_id=org.chat_id,
            session_key=org.session_key,
        )
