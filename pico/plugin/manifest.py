"""Plugin manifest schema.

A manifest is a TOML file (``pico-plugin.toml``) shipped alongside
a plugin's Python package. It declares the plugin's identity,
contribution points, and config schema — everything the registry needs
to know without importing the plugin's code.

The single root table is ``[plugin]``. Contribution arrays are
``[[plugin.contributes.<kind>]]``. Per design, the only contribution
kind that lands in PG-1 is ``memory_backends``; the model accepts
unknown extras silently so future kinds don't break old hosts.

Validation rules worth flagging:

- ``id`` and ``version`` are required (``min_length=1``).
- ``factory`` must look like ``module.path:callable_name`` — checked
  here so a typo fails at startup rather than at first activation.
- Contribution name uniqueness *within a manifest* is enforced; the
  registry separately enforces uniqueness *across* manifests.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# 工厂引用格式为 ``module.path:callable``。正则有意保持宽松：
# 非空的模块式路径、一个冒号以及非空的近似标识符后缀即可。
_FACTORY_REF_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


class _ManifestBase(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class MemoryBackendContribution(_ManifestBase):
    """One ``[[plugin.contributes.memory_backends]]`` entry."""

    name: str = Field(min_length=1)
    factory: str = Field(min_length=1)

    @field_validator("factory")
    @classmethod
    def _factory_is_module_path(cls, v: str) -> str:
        if not _FACTORY_REF_RE.match(v):
            raise ValueError(
                f"factory must be 'module.path:callable', got {v!r}",
            )
        return v


class ToolContribution(_ManifestBase):
    """One ``[[plugin.contributes.tools]]`` entry.

    ``factory`` is a ``module.path:callable`` resolving to a
    ``Callable[[PluginContext], Tool]`` — it returns a single
    :class:`~pico.agent.tools.base.Tool` the host registers into the
    agent's tool set at boot. One tool per entry; a plugin exposing
    several tools lists several entries.
    """

    name: str = Field(min_length=1)
    factory: str = Field(min_length=1)

    @field_validator("factory")
    @classmethod
    def _factory_is_module_path(cls, v: str) -> str:
        if not _FACTORY_REF_RE.match(v):
            raise ValueError(
                f"factory must be 'module.path:callable', got {v!r}",
            )
        return v


class Contributes(_ManifestBase):
    """All contribution arrays for a single manifest.

    ``memory_backends`` and ``tools`` are consumed today; the model keeps
    extra fields silently so future contribution types don't break
    older hosts reading newer manifests.
    """

    memory_backends: list[MemoryBackendContribution] = Field(default_factory=list)
    tools: list[ToolContribution] = Field(default_factory=list)


class PluginManifest(_ManifestBase):
    """Parsed ``pico-plugin.toml``.

    Constructed via :meth:`from_toml_path` / :meth:`from_toml_str`; the
    raw ``__init__`` works too for programmatic tests.
    """

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    display_name: str | None = None
    pico: str | None = None
    bundled: bool = False
    enabled_by_default: bool = False
    contributes: Contributes = Field(default_factory=Contributes)
    config_schema: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _contribution_names_unique(self) -> "PluginManifest":
        # 唯一性只在同一 kind 内约束；后端和工具可以同名，因为占用不同槽位。
        # registry 另行约束跨 manifest 的唯一性。
        for kind, items in (
            ("memory_backend", self.contributes.memory_backends),
            ("tool", self.contributes.tools),
        ):
            names = [c.name for c in items]
            if len(names) != len(set(names)):
                dupes = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"duplicate {kind} name(s) in manifest {self.id!r}: {dupes}",
                )
        return self

    # ── 构造方法 ───────────────────────────────────────────────────

    @classmethod
    def from_toml_str(cls, data: str) -> "PluginManifest":
        """Parse from a raw TOML string."""
        raw = tomllib.loads(data)
        return cls._from_raw(raw)

    @classmethod
    def from_toml_path(cls, path: Path) -> "PluginManifest":
        """Parse from a file on disk.

        Raises ``FileNotFoundError`` if missing, ``tomllib.TOMLDecodeError``
        on malformed TOML, and ``pydantic.ValidationError`` on schema
        mismatch.
        """
        with path.open("rb") as f:
            raw = tomllib.load(f)
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> "PluginManifest":
        # manifest 把所有内容嵌套在 [plugin] 下。交给 pydantic 前先解包，
        # 使 schema 使用相对于 plugin 的字段。
        if "plugin" not in raw:
            raise ValueError(
                "manifest missing top-level [plugin] table",
            )
        return cls.model_validate(raw["plugin"])


__all__ = [
    "Contributes",
    "MemoryBackendContribution",
    "PluginManifest",
    "ToolContribution",
]
