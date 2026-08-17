"""Configuration module for Pico.

This package exposes two layers:

    Base layer (agent runtime):
        ``Config`` + ``load_config`` + path helpers — the fields inherited
        from the base agent framework (agents, channels, providers, tools).

    Pico feature layer:
        ``PicoConfig`` + ``load_pico_config`` + per-feature blocks
        (``ContextConfig``, ``CallEfficiencyConfig``,
        ``SkillForgeConfig``). Defined in :mod:`pico.config.pico`.
"""

from pico.config.loader import get_config_path, load_config
from pico.config.paths import (
    get_cli_history_path,
    get_cron_dir,
    get_data_dir,
    get_logs_dir,
    get_media_dir,
    get_runtime_subdir,
    get_workspace_path,
)
from pico.config.pico import (
    BudgetPolicyConfig,
    CallEfficiencyConfig,
    ContextConfig,
    PicoConfig,
    SkillForgeConfig,
    SmartRoutingConfig,
    TokenWiseConfig,
    ToolResultLifecycleConfig,
    load_pico_config,
)
from pico.config.schema import Config

__all__ = [
    # 基础层
    "Config",
    "load_config",
    "get_config_path",
    "get_data_dir",
    "get_runtime_subdir",
    "get_media_dir",
    "get_cron_dir",
    "get_logs_dir",
    "get_workspace_path",
    "get_cli_history_path",
    # Pico 功能层
    "PicoConfig",
    "load_pico_config",
    "ContextConfig",
    "CallEfficiencyConfig",
    "TokenWiseConfig",
    "SkillForgeConfig",
    "BudgetPolicyConfig",
    "SmartRoutingConfig",
    "ToolResultLifecycleConfig",
]
