"""Read access to registered Local Skills."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pico.agent.tools.base import Tool

if TYPE_CHECKING:
    from pico.memory_engine.skill_forge import LocalSkillCatalog


class SkillReadTool(Tool):
    def __init__(self, catalog: "LocalSkillCatalog") -> None:
        self._catalog = catalog

    @property
    def name(self) -> str:
        return "skill_read"

    @property
    def description(self) -> str:
        return "Read a registered Local Skill by name, including its resolved resource paths."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The exact Local Skill name",
                }
            },
            "required": ["name"],
        }

    async def execute(self, name: str, **_kwargs: Any) -> str:
        content = self._catalog.load_skills_for_context([name], max_inject=1)
        if not content:
            return f"Error: Skill not found: {name}"
        return content


__all__ = ["SkillReadTool"]
