"""Pico product identity and persistent-state roots."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

PRODUCT_NAME = "Pico"
PRODUCT_LOGO = "✦"
CLI_NAME = "pico"
DISTRIBUTION_NAME = "pico-harness"
GLOBAL_STATE_DIRNAME = ".pico"
WORKSPACE_STATE_DIRNAME = ".pico"
DEFAULT_WORKSPACE_SPEC = "~/.pico/workspace"


def get_product_home() -> Path:
    override = os.environ.get("PICO_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / GLOBAL_STATE_DIRNAME


def get_default_workspace() -> Path:
    return get_product_home() / "workspace"


def get_workspace_state_dir(workspace: Path) -> Path:
    return workspace / WORKSPACE_STATE_DIRNAME


def get_project_state_dir(workspace: Path) -> Path:
    resolved = workspace.expanduser().resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved.name).strip(".-") or "workspace"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return get_product_home() / "projects" / f"{slug}-{digest}"
