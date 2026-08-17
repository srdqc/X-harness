"""LocalSkillCatalog — the single owner of the local skill pool.

Absorbs what used to be ``SkillService``: it builds the
:class:`SkillRegistry` + :class:`LocalPool`, runs the SKILL.md file
watcher, and renders skills for the prompt (always-skills, injection,
XML summary).

Retrieval is **not** here — that lives in :class:`LocalSkillSource`
(which reuses this catalog's ``pool`` + ``registry``) and is fused
with the remote sources by :class:`SkillForgeRouter`. The old
``SkillService.select`` / LLM-gate / query-rewriter retrieval path
was retired when the router replaced it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pico.memory_engine.skill_forge.refs import resolve_refs
from pico.memory_engine.skill_local.local_pool import LocalPool
from pico.memory_engine.skill_local.registry import SkillRegistry
from pico.memory_engine.skill_local.types import SkillMeta

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pico.memory_engine.skill_local.watcher import SkillFileWatcher
    from pico.providers.base import LLMProvider


class LocalSkillCatalog:
    """File access + directory rendering over the local skill pool."""

    def __init__(
        self,
        workspace: Path,
        config: Any = None,
        builtin_skills_dir: Path | None = None,
        llm_provider: "LLMProvider | None" = None,  # 为兼容调用方而保留，当前未使用
        *,
        start_watcher: bool = True,
    ):
        # R1：从 config.local_dirs 构建 extra_dirs。列表顺序即优先级，名称冲突时后者覆盖前者。
        # 每个元组格式为 (path, display_name, always_enabled)。
        extra_dirs: list[tuple[Path, str, bool]] = []
        if config is not None and getattr(config, "enabled", False):
            seen_names: dict[str, int] = {}
            for entry in getattr(config, "local_dirs", []) or []:
                entry_path = Path(os.path.expanduser(getattr(entry, "path", "") or "")).resolve()
                entry_enabled = getattr(entry, "enabled", True)
                if not entry_enabled or not entry_path or not entry_path.is_dir():
                    if entry_enabled and getattr(entry, "path", ""):
                        log.warning(
                            "local_dirs: path does not exist or is not a directory: %s",
                            entry_path,
                        )
                    continue
                entry_name = getattr(entry, "name", None) or entry_path.name
                seen_names[entry_name] = seen_names.get(entry_name, 0) + 1
                if seen_names[entry_name] > 1:
                    entry_name = f"{entry_name}_{seen_names[entry_name]}"
                entry_always_enabled = getattr(entry, "always_enabled", True)
                extra_dirs.append((entry_path, entry_name, entry_always_enabled))

        self._registry = SkillRegistry(
            workspace,
            builtin_skills_dir=builtin_skills_dir,
            extra_dirs=extra_dirs,
            scan_max_depth=int(getattr(config, "scan_max_depth", 5) if config else 5),
        )

        self._config = config

        # 对基于文件的 Skill（工作区和内置）执行本地池 BM25 检索，不需模型或 GPU，始终可用。
        self._local_pool = LocalPool(self._registry)

        # 后台 SKILL.md 监视器默认自动启动，使长生命周期消费方 ContextBuilder 能自动获取
        # 对 ``<workspace>/skills/**/SKILL.md`` 的手动编辑。监视器运行在守护线程中，进程退出时自动清理；
        # 缺少 ``watchfiles`` 时退化为空操作并记录一条 INFO 日志。
        #
        # 短生命周期消费方（单个 CLI 命令、为子 Agent 执行一次的 ``build_skills_summary()``）
        # 应传入 ``start_watcher=False``。启动约耗时 25 毫秒，且监视器的守护线程会通过
        # ``on_change`` 绑定方法持有反向强引用，使目录在单次使用后仍存活。对会启动
        # 许多子 Agent 的长生命周期父进程而言，这是真实的线程和句柄泄漏。
        self._file_watcher: "SkillFileWatcher | None" = None
        if start_watcher:
            self.start_file_watcher()

    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    @property
    def pool(self) -> LocalPool:
        return self._local_pool

    def invalidate_skill_cache(self, source: str | None = None) -> None:
        """Invalidate the file-registry cache and refresh the local BM25 index.

        When ``source`` is provided, only that source's slice is
        rebuilt and merged back inside the registry — saves the cost of
        re-walking the unchanged builtin / workspace / external trees.
        Pass ``None`` for a hard reset (rare, only when something off-band
        has rewritten multiple sources at once).

        After the registry update, the local BM25 index is rebuilt
        eagerly so file-watcher events flow straight through to retrieval
        — ``search`` callers never pay the index build on the hot path.
        """
        if source is None:
            self._registry.invalidate_cache()
        else:
            self._registry.invalidate_source(source)
        self._local_pool.rebuild_index()

    def start_file_watcher(self) -> bool:
        """Start the background SKILL.md filesystem watcher.

        Called once automatically from :meth:`__init__`, so consumers
        normally never invoke it directly — it's still public for tests
        and for the rare case of restarting the watcher after an
        external ``stop_file_watcher`` call.

        Idempotent: the first call wires up a daemon thread
        (``watchfiles``-backed) that pipes per-source invalidations into
        :meth:`invalidate_skill_cache` whenever a SKILL.md is added,
        edited or removed under ``<workspace>/skills``. Subsequent calls
        are no-ops and return ``False``.

        Returns ``True`` only when a new watcher thread is now running.
        Returns ``False`` — and the rest of the catalog still works in
        manual-invalidation mode — when:

          - ``watchfiles`` is missing from the install (it's a declared
            dependency, so this only happens in stripped / partial
            installs),
          - the workspace skills directory does not exist,
          - a watcher is already running.

        Builtin / external layers are intentionally **not** watched:
        they are read-only mirrors in this codebase, and the builtin
        layer can carry ~80K files — recursive watching would blow past
        Linux's default inotify watch limit.

        Never raises.
        """
        if self._file_watcher is not None:
            return False
        from pico.memory_engine.skill_local.watcher import SkillFileWatcher

        watcher = SkillFileWatcher(
            roots=[self._registry.workspace_skills],
            on_change=self.invalidate_skill_cache,
            resolve_source=self._registry.resolve_source_for_path,
        )
        if not watcher.start():
            # 启动失败（缺少依赖或根目录）时不设置 ``_file_watcher``，使后续调用能在
            # 前置条件修复后重试，例如工作区已实体化。
            return False
        self._file_watcher = watcher
        return True

    def stop_file_watcher(self) -> None:
        """Signal the watcher thread to exit and best-effort join.

        Safe to call when no watcher was ever started.
        """
        watcher = self._file_watcher
        if watcher is None:
            return
        watcher.stop()
        self._file_watcher = None

    # ------------------------------------------------------------------
    # 旧版 ``SkillsLoader`` API（签名兼容的直接替代）
    # ------------------------------------------------------------------

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """All skills as legacy-shape dicts ``{name, path, source}``."""
        metas = self._registry.list_all()
        if filter_unavailable:
            metas = [m for m in metas if self._registry.check_available(m.name, source=m.source)]
        return [{"name": m.name, "path": str(m.path), "source": m.source} for m in metas]

    def load_skill(self, name: str) -> str | None:
        """Full SKILL.md content, or ``None`` if absent."""
        return self._registry.get_body(name)

    def get_always_skills(self) -> list[SkillMeta]:
        """Skills flagged ``always: true`` whose requirements are met.

        R3: truncation order is by local_dirs list order (which maps to
        source priority in the registry) + alphabetical within each
        source. WARN lists dropped skill names.
        """
        if getattr(self._config, "disable_always", False):
            return []
        # 注册表 list_all 已按分层迭代顺序返回 Skill：工作区 → 按顺序的 extra_dirs → 内置；
        # 每层内按发现顺序。再按（来源优先级，名称）稳定排序，使截断结果可预测。
        all_always = [
            m for m in self._registry.list_all() if m.always and self._registry.check_available(m.name, source=m.source)
        ]
        all_always.sort(key=lambda m: (m.source, m.name))
        cap = int(getattr(self._config, "always_max", 5) or 5)
        if len(all_always) > cap:
            kept = all_always[:cap]
            dropped = all_always[cap:]
            log.warning(
                "always skills (%d) exceed always_max (%d), dropped: %s",
                len(all_always),
                cap,
                ", ".join(m.name for m in dropped),
            )
            return kept
        return all_always

    def load_skills_for_context(
        self,
        skills: "list[SkillMeta] | list[str]",
        max_inject: int | None = None,
    ) -> str:
        """Render the given skills' bodies (frontmatter already stripped) into one blob.

        The body comes straight from ``meta.content`` after registry loading.

        ``max_inject`` caps the number of skills inlined (each body is
        typically 1-5K tokens). When None, falls back to
        ``config.inject_max``. 0 / None disables the cap.

        ``{baseDir}`` placeholders in skill body are substituted with the
        skill's *directory* (``meta.path.parent``, OpenSpace convention)
        so relative references like ``{baseDir}/scripts/foo.py`` resolve
        at runtime — only applies when ``meta.path`` is a real on-disk
        SKILL.md (i.e. filesystem-backed skills; db-only skills without a
        physical path leave the placeholder unchanged).
        """
        if max_inject is None:
            max_inject = getattr(self._config, "inject_max", 0) or 0
        # 向后兼容：接受名称列表，并通过注册表解析。
        if skills and isinstance(skills[0], str):
            resolved: list[SkillMeta] = []
            for name in skills:
                meta = self._lookup_meta(name, None)
                if meta is not None:
                    resolved.append(meta)
            skills = resolved
        parts: list[str] = []
        for m in skills:
            if not m.content:
                continue
            body = m.content
            path_obj = getattr(m, "path", None)
            path_str = str(path_obj) if path_obj is not None else ""
            has_real_path = path_obj is not None and not path_str.startswith("sqlite:")
            header = f"### Skill: {m.name}\n\n"
            if has_real_path and path_obj.parent.exists():
                base_dir = str(path_obj.parent)
                base_dir_refs = body.count("{baseDir}")
                body, _ = resolve_refs(body, path_obj.parent)
                resolved_base_dir = body.count("{baseDir}") < base_dir_refs
                if base_dir_refs == 0 or resolved_base_dir:
                    header = (
                        f"### Skill: {m.name}\n"
                        f"**Skill directory**: `{base_dir}`\n"
                        "Relative refs (e.g. `references/x.md`, `./scripts/y.sh`) "
                        "resolve under this directory — use the absolute form for "
                        "read_file / exec.\n\n"
                    )
            elif not has_real_path:
                body, _ = resolve_refs(body, None)
            parts.append(f"{header}{body}")
            if max_inject and len(parts) >= max_inject:
                break
        return "\n\n---\n\n".join(parts) if parts else ""

    def get_skill_metadata(self, name: str) -> dict | None:
        """Top-level frontmatter dict, or ``None`` if absent."""
        return self._registry.get_raw_metadata(name)

    def build_skills_summary(self, only: "list[SkillMeta] | list[str] | None" = None) -> str:
        """XML-formatted skill directory.

        Args:
            only: When provided, only these skills are included. Accepts
                either ``list[SkillMeta]`` (canonical) or ``list[str]`` of
                skill names (backward-compat for older callers). ``None``
                preserves legacy behavior (full local directory), used as
                a fallback when no selector picked.

        Local skills render via their on-disk ``meta.path``; LLM reads the
        full body via ``read_file``. ``available`` reflects ``requires``
        checks on the registry.
        """
        if only is None:
            metas: list[SkillMeta] = self._registry.list_all()
        else:
            # 向后兼容：接受 Skill 名称列表，通过注册表解析为 SkillMeta，并丢弃未知项。
            if only and isinstance(only[0], str):
                resolved: list[SkillMeta] = []
                for name in only:
                    meta = self._lookup_meta(name, None)
                    if meta is not None:
                        resolved.append(meta)
                only = resolved
            metas = list(only)
        if not metas:
            return ""

        lines: list[str] = ["<skills>"]
        for m in metas:
            name_x = _escape_xml(m.name)
            desc_x = _escape_xml(m.description or m.name)
            available = self._registry.check_available(m.name, source=m.source)
            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{name_x}</name>")
            lines.append(f"    <description>{desc_x}</description>")
            lines.append(f"    <location>{m.path}</location>")
            if not available:
                missing = self._registry.get_missing_requirements(
                    m.name,
                    source=m.source,
                )
                if missing:
                    lines.append(f"    <requires>{_escape_xml(missing)}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def gather_all_skills(self) -> list[SkillMeta]:
        """All skills visible to this catalog — the local file registry
        (workspace, builtin, external mirrors).

        Used by CLI ``skill list`` / inspection helpers — gathers
        everything unranked. Hot-path retrieval goes through
        :class:`SkillForgeRouter` instead.
        """
        return self._registry.list_all()

    def _lookup_meta(
        self,
        name: str,
        source: str | None,
    ) -> SkillMeta | None:
        """Resolve a (name, source) to a ``SkillMeta`` via the local registry."""
        return self._registry.get(name, source=source)


# ----------------------------------------------------------------------
# 渲染辅助函数（模块级、无状态）
# ----------------------------------------------------------------------


def _escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


__all__ = ["LocalSkillCatalog"]
