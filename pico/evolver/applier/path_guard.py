"""Path guard for evolver-generated patches (spec §22 + §22.7).

The immutable set is **evolver-immutable**, not human-immutable.
Human-driven development (PR review) can still touch these paths
through normal workflow; only auto-evolver patches are blocked.

Two pattern types are supported in ``IMMUTABLE_PATTERNS``:

- **Exact file** — e.g. ``"pico/agent/loop/main.py"``. Matches
  exactly this repo-relative path.
- **Directory subtree** — trailing-slash, e.g. ``"pico/evolver/"``.
  Matches the directory itself and any descendant path.

``MUTABLE_OVERRIDES`` takes precedence: a path matching an override is
mutable even if it also matches an immutable pattern. Use this to carve
out mutable sub-trees from a broader immutable directory.

Why repo-relative-string matching instead of full glob:
This module's job is fast, predictable, easy-to-audit gating. Glob
patterns (``**``, ``*.py``) invite surprises (case-sensitivity, slash
behaviour). Exact paths + directory subtrees cover everything in
§22.2 without ambiguity.

Reference layers (spec §22.1, §22.2):
    L1 — Self-reference (evolver/**)
    L2 — Evaluation substrate (eval_engine + external grader)
    L3 — Capability contract (agent loop, tools framework, providers,
         skill loader, sandbox, config schema, ...)
    L4 — Audit / data integrity (tool_audit_hook)
    L5 — Tests, deps, CI
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# 不可变模式列表（规范第 22.2 节）
# ---------------------------------------------------------------------------


# 所有路径均相对于仓库并使用正斜杠分隔；末尾斜杠表示该目录及其全部内容。
IMMUTABLE_PATTERNS: tuple[str, ...] = (
    # ── L1——自指 ─────────────────────────────────────────────────────────
    "pico/evolver/",
    # ── L2——评测基础设施 ─────────────────────────────────────────────────
    "pico/eval_engine/engine.py",
    "pico/eval_engine/adapter/",
    "pico/eval_engine/judge/",
    # AppWorld 演化胶水层（adapter/eval/diagnose/editor/precheck，以及执行 /evaluate 调用、
    # 成功/基础设施分类和结果写入的 grade.py）负责给候选项评分。它位于设计器白名单树内部，
    # 因此防护必须把它排除，对应上游 evaluation/ 入口。
    "benchmarks/appworld/evolve/",
    # 批量评分器编排试验并记录运行器级基础设施；允许候选项编辑它就等于允许改变自己的分母。
    # 可编辑的智能体表面是 agent_cli.py（循环/提示）和 tool.py。
    "benchmarks/appworld/batch.py",
    # ── L3——能力契约 ─────────────────────────────────────────────────────
    "pico/agent/loop/main.py",
    "pico/agent/context/",
    "pico/agent/tools/base.py",
    "pico/agent/tools/registry.py",
    "pico/agent/personalizer/",
    "pico/agent/subagent/",
    # Pico 使用 spine 代替上游智能体 API；运行器是每轮必经的桥梁，属于同一契约层级。
    "pico/agent/spine_runner.py",
    "pico/spine/",
    "pico/providers/",
    "pico/session/",
    "pico/memory_engine/",
    "pico/context_engine/",
    "pico/sandbox/",
    "pico/security/",
    # 配置：schema/loader（.py）不可变，值文件（.yaml/.json）可变。
    "pico/config/__init__.py",
    "pico/config/pico.py",
    "pico/config/loader.py",
    "pico/config/paths.py",
    "pico/config/schema.py",
    "pico/config/update.py",
    "pico/config/update_channels.py",
    "pico/config/update_providers.py",
    # ── L4——审计/数据完整性 ──────────────────────────────────────────────
    "pico/eval_engine/hooks/tool_audit_hook.py",
    # ── L5——测试、依赖和 CI ──────────────────────────────────────────────
    "tests/",
    "pyproject.toml",
    "uv.lock",
    ".github/",
    # ── 核心版本（规范第 22.5 节）——仅人工递增 ───────────────────────────
    "pico/__init__.py",
)


# 不可变子树中仍应保持可变的路径例外。
MUTABLE_OVERRIDES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


class ImmutablePathError(ValueError):
    """Raised when an evolver patch targets an immutable kernel path.

    See spec §22 for the evolver-immutable kernel definition. If a
    legitimate patch is being blocked, two options:

    1. Reframe the patch — split off the mutable part, drop the
       immutable part. This is what `path_guard` expects.
    2. Walk the human-driven development path: open a PR that
       updates ``MUTABLE_OVERRIDES`` (if the path was misclassified)
       or that modifies the kernel directly (then bump ``core_version``
       per spec §22.5).
    """


class UnsafePathError(ImmutablePathError):
    """Raised when a patch path is not a safe repository-relative path."""


# ---------------------------------------------------------------------------
# 内部辅助方法
# ---------------------------------------------------------------------------


def _normalise(path: str) -> str:
    """Return one unambiguous repository-relative path."""
    if not isinstance(path, str):
        raise UnsafePathError(f"Patch path must be a string, got {type(path).__name__}")
    if "\x00" in path:
        raise UnsafePathError("Patch path contains a NUL byte")
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if not p:
        raise UnsafePathError("Patch path must not be empty")
    if p.startswith("/") or p.startswith("//") or re.match(r"^[A-Za-z]:/", p):
        raise UnsafePathError(f"Patch path must be repository-relative: {path!r}")
    parts = p.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise UnsafePathError(f"Patch path contains an unsafe component: {path!r}")
    return p


def _assert_no_symlink(path: str, repo_root: Path) -> None:
    root = repo_root.resolve(strict=True)
    cursor = root
    for part in path.split("/"):
        cursor /= part
        if cursor.is_symlink():
            raise UnsafePathError(f"Patch path traverses a symlink: {path!r}")
    resolved = (root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UnsafePathError(f"Patch path escapes the repository: {path!r}") from exc


def _assert_no_tree_symlink(path: str, repo_root: Path, treeish: str) -> None:
    from pico.evolver.tree import git_ops

    try:
        traverses_symlink = git_ops.tree_path_has_symlink(repo_root, treeish, path)
    except git_ops.GitOpError as exc:
        raise UnsafePathError(f"Could not inspect patch path in parent tree {treeish!r}: {path!r}") from exc
    if traverses_symlink:
        raise UnsafePathError(f"Patch path traverses a symlink in {treeish}: {path!r}")


def _match(path: str, pattern: str) -> bool:
    """Match a normalised path against one pattern.

    Two pattern shapes:

    - Trailing slash → directory subtree match (the directory itself
      and any descendant).
    - Otherwise → exact-string match.
    """
    candidate = path.casefold()
    expected = pattern.casefold()
    if expected.endswith("/"):
        prefix = expected.rstrip("/")
        return candidate == prefix or candidate.startswith(expected)
    return candidate == expected


# ---------------------------------------------------------------------------
# 谓词
# ---------------------------------------------------------------------------


def is_immutable(path: str) -> bool:
    """Return True iff ``path`` is in the evolver-immutable kernel.

    Algorithm:

    1. Normalise the path (handle Windows paths + leading ``./``).
    2. If any pattern in ``MUTABLE_OVERRIDES`` matches → return False
       (override wins).
    3. If any pattern in ``IMMUTABLE_PATTERNS`` matches → return True.
    4. Otherwise → return False (path is mutable by default).
    """
    try:
        norm = _normalise(path)
    except UnsafePathError:
        return True
    for pat in MUTABLE_OVERRIDES:
        if _match(norm, pat):
            return False
    for pat in IMMUTABLE_PATTERNS:
        if _match(norm, pat):
            return True
    return False


def check_patch_paths(
    target_files: Iterable[str],
    *,
    repo_root: str | Path | None = None,
    treeish: str | None = None,
) -> list[str]:
    """Return the subset of ``target_files`` that hit immutable paths.

    Useful for evolver code that wants to inspect violations without
    raising — e.g., to route the patch to a TODO markdown
    (spec §21.4.2) instead of attempting to apply it.

    When ``treeish`` is provided, symlink components are inspected in that Git
    tree rather than in the caller's current checkout.
    """
    offenders: list[str] = []
    root = Path(repo_root) if repo_root is not None else None
    if treeish is not None and root is None:
        raise ValueError("treeish requires repo_root")
    for path in target_files:
        try:
            norm = _normalise(path)
            if root is not None:
                if treeish is None:
                    _assert_no_symlink(norm, root)
                else:
                    _assert_no_tree_symlink(norm, root, treeish)
        except (UnsafePathError, OSError, ValueError):
            offenders.append(path)
            continue
        if is_immutable(norm):
            offenders.append(path)
    return offenders


def assert_patch_allowed(
    target_files: Iterable[str],
    *,
    repo_root: str | Path | None = None,
    treeish: str | None = None,
) -> None:
    """Raise :class:`ImmutablePathError` if any target is immutable.

    Use this as the first-line gate in the evolver applier:

    .. code-block:: python

        from pico.evolver.applier import assert_patch_allowed
        assert_patch_allowed([c.target_file for c in patch.components])
        # ... proceed to apply patch only if no error
    """
    offenders = check_patch_paths(
        target_files,
        repo_root=repo_root,
        treeish=treeish,
    )
    if offenders:
        head = offenders[:5]
        more = f" ... and {len(offenders) - 5} more" if len(offenders) > 5 else ""
        raise ImmutablePathError(
            f"Patch targets {len(offenders)} evolver-immutable path(s): "
            f"{head}{more}. The gates, ledgers, and eval glue may never be "
            "edited by a candidate (see IMMUTABLE_PATTERNS in this module)."
        )


__all__ = [
    "IMMUTABLE_PATTERNS",
    "MUTABLE_OVERRIDES",
    "ImmutablePathError",
    "UnsafePathError",
    "assert_patch_allowed",
    "check_patch_paths",
    "is_immutable",
]
