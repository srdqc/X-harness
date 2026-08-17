"""Minimal in-place updates for ~/.pico/config.json.

Unlike ``save_config`` which re-serializes the entire Pydantic model (and
would bake every runtime default back into the file), these helpers read
the raw JSON, patch a small set of fields, and atomically rewrite via
temp-file + rename. Used by ``pico cron config set`` and the
onboarding wizard so the change persists across restarts without
touching unrelated fields.
"""

from __future__ import annotations

import json
import os
import types
import typing
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel
from pydantic.alias_generators import to_camel
from pydantic_core import PydanticUndefined

from pico.config.loader import get_config_path, read_raw_or_raise
from pico.config.schema import CronConfig


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _unwrap_optional(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_model_class(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _annotation_str(annotation: Any) -> str:
    annotation = _unwrap_optional(annotation)
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return "Literal"
    if origin is list:
        args = typing.get_args(annotation)
        return f"list[{_annotation_str(args[0])}]" if args else "list"
    if origin is dict:
        args = typing.get_args(annotation)
        if args and len(args) == 2:
            return f"dict[{_annotation_str(args[0])}, {_annotation_str(args[1])}]"
        return "dict"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _coerce_value(value: Any, annotation: Any) -> Any:
    if not isinstance(value, str):
        return value

    base = _unwrap_optional(annotation)
    if base is bool:
        value_lower = value.strip().lower()
        if value_lower in ("true", "1", "yes", "on"):
            return True
        if value_lower in ("false", "0", "no", "off"):
            return False
        return value
    if base is int:
        try:
            return int(value)
        except ValueError:
            return value
    if base is float:
        try:
            return float(value)
        except ValueError:
            return value

    origin = typing.get_origin(base)
    if origin is list:
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in value.split(",") if item.strip()]
    if origin is dict:
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return value


def _field_default(field_info: Any) -> Any:
    if field_info.default_factory is not None:
        try:
            return field_info.default_factory()
        except Exception:
            return None
    if field_info.default is PydanticUndefined:
        return None
    return field_info.default


def _flatten_instance(instance: BaseModel, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_name in type(instance).model_fields:
        value = getattr(instance, field_name)
        path = f"{prefix}{field_name}"
        if isinstance(value, BaseModel):
            out.update(_flatten_instance(value, prefix=f"{path}."))
        else:
            out[path] = value
    return out


def _walk_nested_path(model_cls: type[BaseModel], dotted_key: str) -> tuple[type[BaseModel], str]:
    segments = dotted_key.split(".")
    cls = model_cls
    for segment in segments[:-1]:
        field_info = cls.model_fields.get(segment)
        if field_info is None:
            raise KeyError(f"Unknown nested field '{segment}' in {cls.__name__}")
        annotation = _unwrap_optional(field_info.annotation)
        if not _is_model_class(annotation):
            raise KeyError(f"Field '{segment}' in {cls.__name__} is not a nested model")
        cls = annotation
    leaf = segments[-1]
    if leaf not in cls.model_fields:
        raise KeyError(f"Unknown field '{leaf}' in {cls.__name__}")
    return cls, leaf


def _set_nested(dotted_key: str, value: Any, target: dict[str, Any]) -> Any:
    segments = dotted_key.split(".")
    cursor = target
    for segment in segments[:-1]:
        child = cursor.get(segment)
        if not isinstance(child, dict):
            child = {}
            cursor[segment] = child
        cursor = child
    previous = cursor.get(segments[-1])
    cursor[segments[-1]] = value
    return previous


def update_cron_config(
    key: str,
    value: Any,
    *,
    config_path: Path | None = None,
) -> Any:
    """Patch a single CronConfig field on-disk.

    Returns the previous raw value (None if absent). Raises ``KeyError`` if
    ``key`` is not a CronConfig field — defensive only; CLI ``_KEY_HANDLERS``
    already validates before reaching here. Type validation of ``value`` is
    the caller's responsibility (CLI parsers handle it).
    """
    if key not in CronConfig.model_fields:
        raise KeyError(f"Unknown cron config key: {key!r}. Supported: {sorted(CronConfig.model_fields)}")
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    cron_section = data.setdefault("cron", {})
    camel_key = to_camel(key)
    prev = cron_section.get(camel_key)
    cron_section[camel_key] = value
    _write_atomic(path, data)
    logger.info("config/update: cron.{} set to {!r} (was {!r})", key, value, prev)
    return prev


def reset_cron_config(*, config_path: Path | None = None) -> None:
    """Remove the entire ``cron`` section from on-disk config.

    Schema defaults (``forward_channels=["*"]`` / ``default_timezone="Asia/Shanghai"``)
    take effect on next load. Stays consistent with the file's "never bake
    defaults to disk" principle.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    removed = data.pop("cron", None)
    _write_atomic(path, data)
    logger.info("config/update: cron section reset (was {!r})", removed)


def set_language(
    language: str,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch the top-level ``language`` on the on-disk config. Returns previous value.

    Set by the onboarding wizard's language screen. Read by the CLI/wizard copy
    (via ``_t``) and injected into the agent's system prompt so replies use the
    chosen language.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    prev = data.get("language")
    data["language"] = language
    _write_atomic(path, data)
    logger.info("config/update: language set to {!r} (was {!r})", language, prev)
    return prev


def set_default_model(
    model: str,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch ``agents.defaults.model`` on the on-disk config. Returns previous value.

    Used by the onboarding wizard after the user picks a provider: the wizard
    needs to swap the default model to one that matches the chosen provider
    (otherwise ``pico run`` would still route to whatever the freshly
    created ``Config()`` baked in, which is typically a different vendor).
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    defaults = data.setdefault("agents", {}).setdefault("defaults", {})
    prev = defaults.get("model")
    defaults["model"] = model
    _write_atomic(path, data)
    logger.info("config/update: default model set to {} (was {})", model, prev)
    return prev


def set_sandbox_backend(
    backend: str,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch ``sandbox.backend`` on the on-disk config. Returns previous value.

    Used by the onboarding wizard's run-location step. ``backend`` must be one
    of ``SandboxConfig``'s literal values (``none`` / ``auto`` / ``boxlite``);
    the loader validates on next read.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    # sandbox 位于 tools 下（Config.tools.sandbox），而不是根节点。根 Config
    # 禁止额外字段，因此顶层 "sandbox" 键会在下次加载时导致 schema 校验失败。
    section = data.setdefault("tools", {}).setdefault("sandbox", {})
    prev = section.get("backend")
    section["backend"] = backend
    _write_atomic(path, data)
    logger.info("config/update: tools.sandbox.backend set to {!r} (was {!r})", backend, prev)
    return prev


def init_extension_block_defaults(*, config_path: Path | None = None) -> None:
    """Seed the user-facing subset of the memory / plugins / skillForge
    extension blocks into a fresh ``~/.pico/config.json``.

    Called once by the onboarding bootstrap so a new config shows these knobs
    at their schema defaults — discoverable and editable without reading the
    source. Each field is only written when absent (``setdefault``), so this is
    idempotent and never clobbers a value the user (or an earlier wizard step)
    already set. ``memory.backend`` is seeded to its schema default
    (``"myna"``). The onboarding flow keeps that selection and directs
    operators to initialize Myna in the configured Workspace.

    Defaults are pulled from the Pydantic models so this seed can't drift from
    the schema. Plugin configuration stays empty because Myna owns
    repository, profile, runtime-root, and credential selection.

    The optional service fields on ``SkillForgeConfig`` (``embedding_url`` /
    ``embedding_api_key`` / ``reranker_url`` / ``reranker_api_key`` /
    ``mass_library_db``) are deliberately NOT written. They stay at public
    schema defaults and deployments that need hosted services add explicit
    values by hand.

    Key casing follows each block's convention: ``memory`` / ``skillForge`` use
    camelCase (the file-level alias); ``plugins.config`` is a verbatim
    pass-through dict whose keys stay snake_case (each plugin owns its schema).
    """
    from pico.config.pico import (
        MemoryConfig,
        PluginsConfig,
        SkillForgeRouterConfig,
    )

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)

    mem = MemoryConfig()
    memory = data.setdefault("memory", {})
    memory.setdefault("backend", mem.backend)
    memory.setdefault("userId", mem.user_id)
    memory.setdefault("memoryTopK", mem.memory_top_k)

    plugins = data.setdefault("plugins", {})
    plugins.setdefault("disabled", list(PluginsConfig().disabled))
    plugins.setdefault("config", {})

    router_defaults = SkillForgeRouterConfig()
    skill_forge = data.setdefault("skillForge", {})
    skill_forge.setdefault("enabled", True)
    router = skill_forge.setdefault("router", {})
    router.setdefault("enabled", router_defaults.enabled)

    _write_atomic(path, data)
    logger.info("config/update: seeded memory/plugins/skillForge extension defaults")


def set_memory_backend(
    backend: str | None,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch ``memory.backend`` on the on-disk config. Returns previous value.

    ``"myna"`` selects repository Memory; ``None`` disables implicit
    Memory while preserving Sessions and Local Skills.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    section = data.setdefault("memory", {})
    prev = section.get("backend")
    section["backend"] = backend
    _write_atomic(path, data)
    logger.info("config/update: memory.backend set to {!r} (was {!r})", backend, prev)
    return prev


__all__ = [
    "update_cron_config",
    "reset_cron_config",
    "set_default_model",
    "set_sandbox_backend",
    "set_memory_backend",
    "init_extension_block_defaults",
]
