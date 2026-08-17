"""Configuration loading utilities."""

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from pico.config.schema import Config
from pico.product import get_product_home

# Pico 扩展配置块键名的唯一事实来源。
# _migrate_config（在基础 Config 校验前 pop）和 load_pico_config
# （提取到 overrides）都引用这里。新增扩展块只需改这一处，避免重复。
EXTENSION_KEYS = (
    "context",
    "callEfficiency",
    "call_efficiency",
    "tokenWise",
    "skillForge",
    "token_wise",
    "skill_forge",
    # CFG-1 新增项：每个键同时列出配置文件偏好的 camelCase 和 Python
    # 偏好的 snake_case。
    "plugins",
    "memory",
    # Bug2 / runtime-discipline 第五支柱：checkpoint 策略等。
    "runtime",
    # 仓库内可观测性 tracing（pico.tracing）。
    "tracing",
)

# 保存当前配置路径的全局变量，用于多实例支持。
_current_config_path: Path | None = None

_REMOVED_CHANNELS = frozenset(
    {"telegram", "slack", "discord", "whatsapp", "matrix", "mochat", "dingtalk", "email", "weixin"}
)


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return get_product_home() / "config.json"


class ConfigReadError(Exception):
    """An existing config file could not be parsed. Callers doing a
    read-modify-write MUST NOT proceed: overwriting would replace the user's
    whole config with just their section (data loss). Only a genuinely-absent
    file is safe to create fresh.

    Deliberately NOT a RuntimeError: the CLI write commands wrap their ops in a
    broad ``except RuntimeError`` (for provider OAuth-refusal etc.), and we want
    a parse error to bypass those and reach the single ``run()`` handler (or a
    caller's explicit ``except ConfigReadError``), not be swept up implicitly."""


def _reject_unsupported_config(data: dict[str, Any]) -> None:
    if "sentinel" in data:
        raise ValueError(
            "Config contains sentinel, which is no longer supported; remove that section before starting Pico."
        )
    gateway = data.get("gateway")
    if isinstance(gateway, dict) and "heartbeat" in gateway:
        raise ValueError(
            "Config contains gateway.heartbeat, which is no longer supported; remove that section before starting Pico."
        )
    channels = data.get("channels")
    removed = sorted(_REMOVED_CHANNELS.intersection(channels)) if isinstance(channels, dict) else []
    if removed:
        name = removed[0]
        raise ValueError(
            f"Config contains channels.{name}, which is no longer supported; remove that section before starting Pico."
        )
    tools = data.get("tools")
    if isinstance(tools, dict) and "media" in tools:
        raise ValueError(
            "Config contains tools.media, which is no longer supported; remove that section before starting Pico."
        )
    if isinstance(tools, dict) and any(key in tools for key in ("deepResearch", "deep_research")):
        raise ValueError(
            "Config contains tools.deepResearch, which is no longer supported; "
            "remove that section before starting Pico."
        )
    for skill_forge_key in ("skillForge", "skill_forge"):
        skill_forge = data.get(skill_forge_key)
        if not isinstance(skill_forge, dict):
            continue
        router = skill_forge.get("router")
        if not isinstance(router, dict):
            continue
        if "hub" in router:
            raise ValueError(
                f"Config contains {skill_forge_key}.router.hub, which is no longer supported; "
                "remove that section before starting Pico."
            )
        weights = router.get("weights")
        if isinstance(weights, dict) and "hub" in weights:
            raise ValueError(
                f"Config contains {skill_forge_key}.router.weights.hub, which is no longer supported; "
                "remove that entry before starting Pico."
            )
    for router_key in ("skillRouter", "skill_router"):
        router = data.get(router_key)
        if not isinstance(router, dict):
            continue
        if "hub" in router:
            raise ValueError(
                f"Config contains {router_key}.hub, which is no longer supported; "
                "remove that section before starting Pico."
            )
        weights = router.get("weights")
        if isinstance(weights, dict) and "hub" in weights:
            raise ValueError(
                f"Config contains {router_key}.weights.hub, which is no longer supported; "
                "remove that entry before starting Pico."
            )


def read_raw_or_raise(path: Path) -> dict[str, Any]:
    """Read a config file as raw JSON for a read-modify-write cycle.

    Returns ``{}`` ONLY when the file is absent. A present-but-unreadable file
    raises :class:`ConfigReadError` rather than returning ``{}`` -- returning
    ``{}`` and then writing was the bug that wiped a real config over a lone
    JSON syntax error (e.g. a // comment). The single read path for every
    ``update_*`` write module.
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}  # 空文件没有数据可丢失，可像文件不存在一样安全重建
        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
        _reject_unsupported_config(data)
        return data
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ConfigReadError(
            f"{path} is not valid JSON ({exc}). Fix it first (JSON allows no comments or "
            "trailing commas); your config was left unchanged."
        ) from exc


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    config: Config | None = None
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
        except json.JSONDecodeError as e:
            # 配置文件损坏时用默认值启动，避免写入中途的短暂竞争让调用方瘫痪；
            # 但必须明确警告，否则持续的语法错误会无声重置所有设置。
            # 若要改为抛错，必须先让 save_config 支持原子写入（另行修改）。
            msg = (
                f"config at {path} is not valid JSON ({e}) -- IGNORING it and running on "
                "DEFAULTS. Fix the file (JSON allows no comments or trailing commas) and restart."
            )
            print(f"WARNING: {msg}", file=sys.stderr)
            logger.warning(msg)
        else:
            try:
                config = Config.model_validate(data)
            except ValidationError as e:
                # schema 不匹配属于用户或程序错误，应明确暴露而不是用默认值掩盖。
                # 静默使用默认值会让“功能 X 为什么没生效”的排查从 24 秒拖到 24 小时。
                raise ValueError(
                    f"Config at {path} fails schema validation:\n{e}",
                ) from e

    if config is None:
        config = Config()

    return config


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict, *, pop_extension_keys: bool = True) -> dict:
    """Migrate old config formats to current.

    ``pop_extension_keys``: when True (default, used by ``load_config``),
    strip extension block keys so the base ``Config(extra='forbid')``
    doesn't reject them. Set to False when the caller needs to read
    extension blocks from the migrated data (``load_pico_config``).
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    _reject_unsupported_config(data)

    # 将 tools.exec.restrictToWorkspace 迁移到 tools.restrictToWorkspace。
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    # 删除已退役的 remembered-Skill 提取字段。该迁移绝不修改 ``memory.backend``；
    # 显式选择旧后端时必须继续进入插件解析，并以包含运维修复指引的错误失败。
    agents = data.get("agents", {})
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    if isinstance(defaults, dict):
        for legacy_key in ("everos", "everosSkillLight", "everos_skill_light"):
            if defaults.pop(legacy_key, None) is not None:
                _log.info("Migrated: dropped agents.defaults.%s (retired)", legacy_key)

    memory = data.get("memory")
    if isinstance(memory, dict):
        for legacy_key in ("agentId", "agent_id"):
            if memory.pop(legacy_key, None) is not None:
                _log.info("Migrated: dropped memory.%s (retired)", legacy_key)

    # skills_dir → local_dirs 迁移现由
    # SkillForgeConfig._migrate_skills_dir model_validator（R5）处理。

    # 把旧版顶层 ``skillRouter`` / ``skill_router`` 块嵌套进
    # ``skillForge.router``。router 现在属于 SkillForge 子块，不再是并列的
    # 顶层键；显式 ``skillForge.router`` 优先。
    router_block = data.pop("skillRouter", None)
    if router_block is None:
        router_block = data.pop("skill_router", None)
    if router_block is not None:
        if isinstance(data.get("skillForge"), dict):
            sf_key = "skillForge"
        elif isinstance(data.get("skill_forge"), dict):
            sf_key = "skill_forge"
        else:
            sf_key = "skillForge"
            data[sf_key] = {}
        sf = data[sf_key]
        if isinstance(sf, dict) and "router" not in sf:
            sf["router"] = router_block
            _log.info("Migrated: top-level skillRouter → skillForge.router")

    retired_skill_forge_fields = {
        "everos",
        "evolveModel",
        "evolve_model",
        "detectModel",
        "detect_model",
        "detectMinToolCalls",
        "detect_min_tool_calls",
        "autoDetect",
        "auto_detect",
        "autoEvolve",
        "auto_evolve",
        "evolveTriggerSuccessRate",
        "evolve_trigger_success_rate",
        "evolveTriggerMinInvocations",
        "evolve_trigger_min_invocations",
        "draftFirstActivation",
        "draft_first_activation",
        "retirementIdleDays",
        "retirement_idle_days",
    }
    for sf_key in ("skillForge", "skill_forge"):
        sf = data.get(sf_key)
        if not isinstance(sf, dict):
            continue
        for legacy_key in retired_skill_forge_fields:
            if sf.pop(legacy_key, None) is not None:
                _log.info("Migrated: dropped %s.%s (retired)", sf_key, legacy_key)
        router = sf.get("router")
        if isinstance(router, dict):
            for legacy_key in (
                "mass",
                "weights",
                "overFetchFactor",
                "over_fetch_factor",
                "dedupBy",
                "dedup_by",
            ):
                if router.pop(legacy_key, None) is not None:
                    _log.info("Migrated: dropped %s.router.%s (retired)", sf_key, legacy_key)

    # ── 基础 Config 校验前弹出扩展键 ────────────────────────────
    if pop_extension_keys:
        for ek in EXTENSION_KEYS:
            data.pop(ek, None)

    return data
