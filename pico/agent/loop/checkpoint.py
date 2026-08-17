"""Per-turn shadow-git checkpoint of the workspace (Bug2 safety net).

Commits the workspace to an out-of-band git repo (separate ``--git-dir``,
work-tree pointed at the real workspace) at the end of each turn. The user's
own ``.git`` is never touched. A truncated multi-file edit therefore leaves a
recoverable snapshot, and the interrupted turn's changed files can be listed
for the next turn's recovery prompt.

Scope (documented limits):
- Only filesystem state is snapshotted — not conversation state.
- Changes made via shell tools (``rm``/``mv``/``sed -i``) are captured by the
  next ``add -A`` but are not attributable to a specific tool call. This is an
  *undo stack for the working tree*, not full crash recovery.
- Granularity is per-turn (one commit per turn), matching Claude Code/Cursor.

Safety layers (defense in depth against snapshotting things the user doesn't
want stored):
1. ``info/exclude`` ships an expanded default blacklist covering common build
   artifacts, virtualenvs, IDE state, OS junk, and likely-credential paths.
2. The work-tree's own ``.gitignore`` files are honored automatically by git
   (standard ``add -A`` semantics) — so anything the user marked private in
   their own repo stays out of the shadow as well.
3. ``gc.auto`` is configured so a periodic ``git gc --auto`` keeps long-lived
   sessions from accumulating loose objects forever.

Every git invocation is best-effort: failures are logged and degrade to a
no-op (return ``None``) so the checkpoint layer can never break a turn.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from pico.product import WORKSPACE_STATE_DIRNAME

# 将提交者身份写入影子仓库，使提交不依赖也不修改用户的全局 Git 配置。
_GIT_IDENT = (
    "-c",
    "user.name=Pico",
    "-c",
    "user.email=checkpoint@pico.local",
    "-c",
    "commit.gpgsign=false",
)


# 将临时或高风险模式写入影子仓库的 ``info/exclude``。通过纵深防御，即使工作区
# 没有 ``.gitignore``，它们也不会进入快照。分类与真实项目通常忽略的内容保持一致：
#
# - 影子仓库自身和 Python 缓存：避免递归收录仓库和常见 Python 构建噪声。
# - 构建和打包产物：常见多语言输出目录，可达 GB 级且没有恢复价值。
# - 虚拟环境：体积大，且可从锁文件重建。
# - 凭据和 dotenv：高影响泄漏载体。用户的 ``.gitignore`` 通常会覆盖，但仍需兜底排除。
# - 日志、操作系统垃圾和 IDE 状态：不是密钥，只会使仓库膨胀。
_DEFAULT_EXCLUDES = """\
# Pico shadow-git default excludes (see checkpoint.py).
# Layered on top of any .gitignore files in the work-tree.

# Self + Python caches
.pico/
__pycache__/
*.pyc
*.pyo

# Build / package artifacts
dist/
build/
target/
*.egg-info/
.eggs/
node_modules/
.next/
.nuxt/
out/

# Virtualenvs
# (``env/`` deliberately omitted — too easily collides with a legitimate
# project source dir; users whose env IS a virtualenv typically have it
# in their own .gitignore, which S4-A honors automatically.)
venv/
.venv/
.tox/

# Credentials & dotenv (defense in depth — usually in user's .gitignore too)
.env
.env.*
*.key
*.pem
*.crt
*.p12
.aws/credentials
secrets.yaml
secrets.yml

# Logs
*.log
logs/

# OS junk
.DS_Store
Thumbs.db

# IDE state
.idea/
.vscode/
"""


# 对影子仓库触发 ``git gc --auto`` 的成功提交间隔。``--auto`` 让 Git 根据内部启发式
# 自行判断是否需要 GC，此处只提供心跳。0 完全禁用周期调用。
_GC_EVERY_N_COMMITS = 50


# 单个 Git 子进程的上限。如果没有此限制，NFS 锁、被占用的 ``.git/index.lock`` 或磁盘已满
# 都可能让 ``communicate()`` 永久挂起并锁死 Agent Loop。该值宽裕到可容纳正常冷启动，
# 又足够严格，能在一个 Turn 内识别真正挂起。
_GIT_TIMEOUT_SECONDS = 30.0


class CheckpointService:
    """Shadow-git working-tree snapshots, one commit per turn."""

    def __init__(
        self,
        workspace: Path,
        shadow_dir: str = f"{WORKSPACE_STATE_DIRNAME}/shadow.git",
        *,
        state: Path | None = None,
    ) -> None:
        self._workspace = Path(workspace).expanduser().resolve()
        state_root = Path(state).expanduser().resolve() if state is not None else self._workspace
        shadow_path = Path(shadow_dir)
        if state is not None and shadow_path.parts[:1] == (WORKSPACE_STATE_DIRNAME,):
            shadow_path = Path(*shadow_path.parts[1:])
        candidate = (state_root / shadow_path).resolve()
        root_label = "workspace" if state is None else "state root"
        # 包容性是关键不变式：如果影子 Git 落到状态根目录之外，按工作区的恢复隔离就会失效。
        # 不同工作区中配置了类似逃逸路径的另一个 AgentLoop 可能共享仓库，交叉污染
        # ``edited_files``。``..``、绝对路径、``""`` 和 ``"."`` 都会触发该问题；
        # 应以清晰错误拒绝，而不是让解析后的路径静默漂移。
        if candidate == state_root or not candidate.is_relative_to(state_root):
            raise ValueError(
                f"shadow_dir={shadow_dir!r} must resolve to a path strictly "
                f"under the {root_label} ({state_root}); got {candidate}"
            )
        self._git_dir = candidate
        self._ready = False
        self._commit_count = 0

    async def _git(self, *args: str) -> tuple[int, str, str]:
        """Run a git command against the shadow repo. Returns (rc, out, err).

        ``core.quotePath=false`` keeps non-ASCII paths (CJK/Japanese/emoji) as
        real UTF-8 in output instead of git's default octal-escaped form —
        without this, ``edited_files`` would land in the recovery prompt as
        ``"\\346\\265\\213"`` gibberish.
        """
        cmd = (
            "git",
            f"--git-dir={self._git_dir}",
            f"--work-tree={self._workspace}",
            "-c",
            "core.quotePath=false",
            *args,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(),
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            # NFS、索引锁或磁盘已满时，不泄漏僵尸进程，也不让 Turn 挂起。
            # 合成非零返回码，触发调用方的失败降级路径。
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            logger.debug(
                "checkpoint git timed out after {}s: {}",
                _GIT_TIMEOUT_SECONDS,
                " ".join(args[:2]),
            )
            return -1, "", "timeout"
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")

    async def _ensure_init(self) -> bool:
        """Lazily initialize the shadow repo. Idempotent; returns readiness."""
        if self._ready:
            return True
        try:
            if not (self._git_dir / "HEAD").exists():
                self._git_dir.parent.mkdir(parents=True, exist_ok=True)
                rc, _, err = await self._git("init")
                if rc != 0:
                    logger.debug("checkpoint init failed: {}", err.strip())
                    return False
                # 在影子 Git 旁写入可发现性提示，让注意到 ``.pico/`` 的用户无需搜索代码库即可识别它。
                # 该操作尽力而为，写入失败不影响主流程。
                try:
                    notice = self._git_dir.parent / "NOTICE.txt"
                    notice.write_text(
                        "This directory is created by Pico's runtime "
                        "checkpoint feature (a per-turn safety net). It is "
                        "an out-of-band shadow git repo; your own .git is "
                        "untouched.\n\n"
                        "Safe to delete; will be recreated on next agent run. "
                        'Disable via `runtime.checkpoint.policy = "never"` '
                        "in your Pico config (typically "
                        "~/.pico/config.json, or whichever file you "
                        "passed via --config).\n",
                        encoding="utf-8",
                    )
                except OSError:
                    pass
            # 分层忽略：影子仓库默认规则加用户自己的 .gitignore，后者由 Git 在工作树中自动遍历。
            # 两者共同将构建产物、临时状态和用户标记为私密的文件排除在所有快照之外。
            exclude = self._git_dir / "info" / "exclude"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_text(_DEFAULT_EXCLUDES, encoding="utf-8")
            # gc.auto 是 Git 判断对象和引用是否已累积到需要真正 GC 的阈值。初始化时设置一次，
            # 后续每次 ``git gc --auto`` 都可查询同一阈值，无需每次传入 ``-c``。
            await self._git("config", "gc.auto", "256")
            # gc.autoDetach=false 让 Git 决定自动 GC 时在前台运行，而不分离后台守护进程。
            # 分离的 GC 会与工作区清理（测试临时目录、Agent 关闭）竞态；rmtree 遇到仍在写
            # ``objects/`` 的 GC 时会报“目录非空”。同步 GC 受 _GIT_TIMEOUT_SECONDS 约束，同样不会挂起 Turn。
            await self._git("config", "gc.autoDetach", "false")
            self._ready = True
            return True
        except OSError as exc:
            logger.debug("checkpoint init error: {}", exc)
            return False

    async def commit_turn(self, label: str) -> tuple[str | None, list[str]]:
        """Snapshot the current worktree as one commit.

        Returns ``(checkpoint_id, changed_files)``. When nothing changed
        since the last turn, or on any git failure, returns ``(None, [])``.
        """
        if not await self._ensure_init():
            return None, []
        try:
            rc, _, err = await self._git("add", "-A")
            if rc != 0:
                logger.debug("checkpoint add failed: {}", err.strip())
                return None, []
            # 当前 Turn 暂存的文件就是本 Turn 的变更，需在提交前捕获。
            _, out, _ = await self._git("diff", "--cached", "--name-only")
            changed = [ln for ln in out.splitlines() if ln.strip()]
            if not changed:
                return None, []  # 没有可快照的内容
            rc, _, err = await self._git(*_GIT_IDENT, "commit", "-m", label)
            if rc != 0:
                logger.debug("checkpoint commit failed: {}", err.strip())
                return None, []
            rc, out, _ = await self._git("rev-parse", "--short", "HEAD")
            cid = out.strip() or None
            self._commit_count += 1
            await self._maybe_gc()
            return cid, changed
        except OSError as exc:
            logger.debug("checkpoint commit error: {}", exc)
            return None, []

    async def _maybe_gc(self) -> None:
        """Periodic ``git gc --auto`` so long-lived sessions don't accumulate
        loose objects forever. ``--auto`` is a no-op below ``gc.auto`` (256
        loose objects by default), so the cost in steady state is one cheap
        rev-list count, not a real repack.

        ``_commit_count`` is per-instance and resets when CheckpointService
        is re-constructed (e.g. fresh AgentLoop start). The 50-commit
        heartbeat is therefore a hint, not a guarantee — git's own
        ``gc.auto=256`` threshold (set at init) is the load-bearing safety
        net that catches accumulated loose objects across process restarts.
        """
        if _GC_EVERY_N_COMMITS <= 0:
            return
        if self._commit_count % _GC_EVERY_N_COMMITS != 0:
            return
        rc, _, err = await self._git("gc", "--auto")
        if rc != 0:
            logger.debug("checkpoint gc failed: {}", err.strip())


__all__ = ["CheckpointService"]
