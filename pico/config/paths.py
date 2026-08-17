"""Runtime path helpers derived from the active config context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pico.config.loader import get_config_path
from pico.product import DEFAULT_WORKSPACE_SPEC, get_default_workspace, get_product_home, get_project_state_dir
from pico.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from pico.config.schema import Config


@dataclass(frozen=True)
class RuntimePaths:
    workspace: Path
    state: Path


def resolve_foreground_paths(
    config: "Config",
    *,
    workspace: str | None = None,
    cwd: Path | None = None,
) -> RuntimePaths:
    if workspace:
        explicit = Path(workspace).expanduser()
        return RuntimePaths(workspace=explicit, state=explicit)
    configured = Path(config.workspace_path).expanduser()
    if config.agents.defaults.workspace != DEFAULT_WORKSPACE_SPEC:
        return RuntimePaths(workspace=configured, state=configured)
    resolved_workspace = (cwd or Path.cwd()).expanduser().resolve()
    return RuntimePaths(
        workspace=resolved_workspace,
        state=get_project_state_dir(resolved_workspace),
    )


def resolve_service_paths(config: "Config") -> RuntimePaths:
    workspace = Path(config.workspace_path).expanduser()
    return RuntimePaths(workspace=workspace, state=workspace)


def get_data_dir() -> Path:
    """Return the instance-level runtime data directory."""
    return ensure_dir(get_config_path().parent)


def get_runtime_subdir(name: str) -> Path:
    """Return a named runtime subdirectory under the instance data dir."""
    return ensure_dir(get_data_dir() / name)


def get_media_dir(channel: str | None = None) -> Path:
    """Return the media directory, optionally namespaced per channel."""
    base = get_runtime_subdir("media")
    return ensure_dir(base / channel) if channel else base


def get_cron_dir() -> Path:
    """Return the cron storage directory."""
    return get_runtime_subdir("cron")


def get_cache_dir() -> Path:
    """Return the disposable, refetchable on-disk cache directory."""
    return get_runtime_subdir("cache")


def get_sandbox_dir(backend: str) -> Path:
    """Return the sandbox runtime home directory for the given backend.

    e.g. backend='boxlite' → <data_dir>/sandbox/boxlite (used as boxlite's
    home_dir so its DB, images, and layers live under Pico's data dir
    instead of ~/.boxlite).
    """
    return ensure_dir(get_runtime_subdir("sandbox") / backend)


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_runtime_subdir("logs")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure the agent workspace path."""
    path = (
        get_default_workspace()
        if workspace is None or workspace == DEFAULT_WORKSPACE_SPEC
        else Path(workspace).expanduser()
    )
    return ensure_dir(path)


def get_cli_history_path() -> Path:
    """Return the shared CLI history file path."""
    return get_product_home() / ".pico_history"
