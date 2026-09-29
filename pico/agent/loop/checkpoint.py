"""为每个 Turn 创建 Workspace 的 out-of-band shadow-git checkpoint（Bug2 safety net）。

Turn 结束时，Service 用独立 ``--git-dir``、真实 Workspace 作为 work-tree 提交一份 Snapshot，
never 触碰 User 自有 ``.git``。因此多文件编辑在迭代上限处被截断时仍有可恢复现场，Changed
Files 与 Commit id 可进入 Next Turn recovery prompt。Granularity 是 per-turn one commit，与
Claude Code/Cursor 类似。

Scope 有明确上限：只 Snapshot Filesystem state，不保存 Conversation state；Shell Tool 的
``rm``/``mv``/``sed -i`` 会被下一次 ``add -A`` 捕获，却无法归因到具体 Tool call。这是 Working
Tree 的 *undo stack*，not full crash recovery。

Safety 采用 Defense in Depth：Shadow Repo ``info/exclude`` 默认排除 Build、Virtualenv、IDE、OS
Junk 与 likely-credential path；Work-tree 自身 ``.gitignore`` 继续按 Git standard semantics 生效；
``gc.auto`` 与周期 ``git gc --auto`` 控制 loose objects。Every Git invocation 都是 best-effort，
失败记录为显式 ``FAILED`` 结果，Checkpoint 绝不能打断 Turn。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from pico.product import WORKSPACE_STATE_DIRNAME
from pico.utils.atomic_io import StorageCorruptionError, locked_append, locked_read

_CATALOGUE_SCHEMA_VERSION = 1
_CATALOGUE_FILENAME = "checkpoint-catalogue.jsonl"


class CheckpointStatus(str, Enum):
    """Outcome of one checkpoint attempt."""

    CREATED = "created"
    UNCHANGED = "unchanged"
    FAILED = "failed"


@dataclass(frozen=True)
class CheckpointRecord:
    """One durable checkpoint attempt and its recovery correlation facts."""

    record_id: str
    revision: int
    status: CheckpointStatus
    checkpoint_id: str | None
    workspace_path: str
    shadow_git_dir: str
    session_id: str | None
    boundary_id: str | None
    turn_id: str | None
    changed_paths: tuple[str, ...]
    created_at: str
    schema_version: int = _CATALOGUE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "revision": self.revision,
            "status": self.status.value,
            "checkpoint_id": self.checkpoint_id,
            "workspace_path": self.workspace_path,
            "shadow_git_dir": self.shadow_git_dir,
            "session_id": self.session_id,
            "boundary_id": self.boundary_id,
            "turn_id": self.turn_id,
            "changed_paths": list(self.changed_paths),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CheckpointResult:
    """Explicit result while preserving legacy two-value unpacking."""

    status: CheckpointStatus
    checkpoint_id: str | None
    changed_paths: tuple[str, ...]
    record_id: str | None

    def __iter__(self) -> Iterator[object]:
        yield self.checkpoint_id
        yield list(self.changed_paths)

    def __getitem__(self, index: int) -> object:
        return (self.checkpoint_id, list(self.changed_paths))[index]

    def __len__(self) -> int:
        return 2


@dataclass(frozen=True)
class CheckpointValidation:
    """Read-only validation evidence for a catalogue record."""

    usable: bool
    reason: str | None
    record: CheckpointRecord | None = None
    tree_id: str | None = None
    workspace_drifted: bool | None = None
    drifted_paths: tuple[str, ...] = ()


class CheckpointCatalogueError(StorageCorruptionError):
    """The durable checkpoint catalogue cannot be trusted."""

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
    """管理每 Turn 一个 Commit 的 Shadow-git Working-tree Snapshot。

    Workspace 与可选 State Root 在构造时解析；Shadow dir 必须严格位于对应 Root 下，``..``、
    Absolute、空或 ``.`` 等 Escape 立即 `ValueError`，避免不同 Workspace 共享 Shadow Repo 并污染
    ``edited_files``。Repo Lazy Initialize，Commit/GC 计数按 Service instance 维护。

    所有 Git 子进程使用独立 Identity、禁用 GPG、设置 ``core.quotePath=false`` 以保留 CJK Path，
    并受 30 秒 Timeout。Service 只提供恢复证据，不修改 User Git 配置或分支。
    """

    def __init__(
        self,
        workspace: Path,
        shadow_dir: str = f"{WORKSPACE_STATE_DIRNAME}/shadow.git",
        *,
        state: Path | None = None,
        record_id_factory: Callable[[], str] | None = None,
        now_fn: Callable[[], datetime] | None = None,
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
        self._catalogue_path = self._git_dir.parent / _CATALOGUE_FILENAME
        self._record_id_factory = record_id_factory or (lambda: uuid.uuid4().hex)
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._ready = False
        self._commit_count = 0

    @property
    def workspace_path(self) -> Path:
        """Resolved live Workspace identity owned by this checkpoint service."""

        return self._workspace

    @staticmethod
    def _optional_identity(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise CheckpointCatalogueError(f"checkpoint {field} must be null or a non-empty string")
        return value

    @classmethod
    def _decode_record(cls, raw: Any) -> CheckpointRecord:
        if not isinstance(raw, dict):
            raise CheckpointCatalogueError("checkpoint catalogue record must be an object")
        if raw.get("schema_version") != _CATALOGUE_SCHEMA_VERSION:
            raise CheckpointCatalogueError("unsupported checkpoint catalogue schema version")
        record_id = raw.get("record_id")
        revision = raw.get("revision")
        if not isinstance(record_id, str) or not record_id:
            raise CheckpointCatalogueError("checkpoint record_id must be a non-empty string")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise CheckpointCatalogueError("checkpoint revision must be a positive integer")
        try:
            status = CheckpointStatus(raw.get("status"))
        except (TypeError, ValueError) as exc:
            raise CheckpointCatalogueError("checkpoint status is invalid") from exc
        checkpoint_id = cls._optional_identity(raw.get("checkpoint_id"), "checkpoint_id")
        workspace_path = raw.get("workspace_path")
        shadow_git_dir = raw.get("shadow_git_dir")
        created_at = raw.get("created_at")
        changed_paths = raw.get("changed_paths")
        if not isinstance(workspace_path, str) or not workspace_path or not Path(workspace_path).is_absolute():
            raise CheckpointCatalogueError("checkpoint workspace_path must be absolute")
        if not isinstance(shadow_git_dir, str) or not shadow_git_dir or not Path(shadow_git_dir).is_absolute():
            raise CheckpointCatalogueError("checkpoint shadow_git_dir must be absolute")
        if not isinstance(created_at, str) or not created_at:
            raise CheckpointCatalogueError("checkpoint created_at must be a non-empty string")
        try:
            parsed_created_at = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise CheckpointCatalogueError("checkpoint created_at must be ISO-8601") from exc
        if parsed_created_at.tzinfo is None:
            raise CheckpointCatalogueError("checkpoint created_at must include a timezone")
        if not isinstance(changed_paths, list) or any(
            not isinstance(path, str) or not path for path in changed_paths
        ):
            raise CheckpointCatalogueError("checkpoint changed_paths must contain non-empty strings")
        if len(changed_paths) != len(set(changed_paths)):
            raise CheckpointCatalogueError("checkpoint changed_paths contains duplicates")
        session_id = cls._optional_identity(raw.get("session_id"), "session_id")
        boundary_id = cls._optional_identity(raw.get("boundary_id"), "boundary_id")
        turn_id = cls._optional_identity(raw.get("turn_id"), "turn_id")
        if boundary_id is not None and session_id is None:
            raise CheckpointCatalogueError("checkpoint boundary_id requires session_id")
        if status is CheckpointStatus.CREATED:
            if checkpoint_id is None or len(checkpoint_id) not in {40, 64} or any(
                char not in "0123456789abcdef" for char in checkpoint_id
            ):
                raise CheckpointCatalogueError("created checkpoint requires a full Git object id")
            if not changed_paths:
                raise CheckpointCatalogueError("created checkpoint requires changed_paths")
        elif checkpoint_id is not None or changed_paths:
            raise CheckpointCatalogueError(
                "unchanged or failed checkpoint cannot claim an object or changed paths"
            )
        return CheckpointRecord(
            record_id=record_id,
            revision=revision,
            status=status,
            checkpoint_id=checkpoint_id,
            workspace_path=workspace_path,
            shadow_git_dir=shadow_git_dir,
            session_id=session_id,
            boundary_id=boundary_id,
            turn_id=turn_id,
            changed_paths=tuple(changed_paths),
            created_at=created_at,
        )

    @classmethod
    def _decode_catalogue(cls, raw: str) -> dict[str, CheckpointRecord]:
        records: dict[str, CheckpointRecord] = {}
        if not raw:
            return records
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CheckpointCatalogueError(
                    f"invalid checkpoint catalogue JSON at line {line_number}"
                ) from exc
            record = cls._decode_record(decoded)
            previous = records.get(record.record_id)
            if previous is None:
                if record.revision != 1:
                    raise CheckpointCatalogueError("first checkpoint record revision must be 1")
            else:
                if record.revision != previous.revision + 1:
                    raise CheckpointCatalogueError("checkpoint record revisions are not contiguous")
                immutable_fields = (
                    "status",
                    "checkpoint_id",
                    "workspace_path",
                    "shadow_git_dir",
                    "changed_paths",
                    "created_at",
                )
                if any(getattr(record, field) != getattr(previous, field) for field in immutable_fields):
                    raise CheckpointCatalogueError("checkpoint correlation update changed immutable evidence")
                for field in ("session_id", "boundary_id", "turn_id"):
                    old = getattr(previous, field)
                    new = getattr(record, field)
                    if old is not None and new != old:
                        raise CheckpointCatalogueError(
                            f"checkpoint correlation update changed existing {field}"
                        )
            records[record.record_id] = record
        return records

    def _load_catalogue(self) -> dict[str, CheckpointRecord]:
        raw, _epoch, _known = locked_read(self._catalogue_path)
        return self._decode_catalogue(raw or "")

    def _append_record(self, record: CheckpointRecord) -> None:
        self._decode_record(record.to_dict())
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        def validate_append(raw: str) -> None:
            separator = "" if not raw or raw.endswith(("\n", "\r")) else "\n"
            self._decode_catalogue(f"{raw}{separator}{line}\n")

        locked_append(
            self._catalogue_path,
            [line],
            require_existing=record.revision > 1,
            validate_existing=validate_append,
        )

    def _record_result(
        self,
        status: CheckpointStatus,
        *,
        checkpoint_id: str | None = None,
        changed_paths: tuple[str, ...] = (),
        session_id: str | None,
        boundary_id: str | None,
        turn_id: str | None,
    ) -> CheckpointResult:
        record_id = self._record_id_factory()
        if not isinstance(record_id, str) or not record_id:
            logger.debug("checkpoint catalogue id factory returned an invalid id")
            return CheckpointResult(CheckpointStatus.FAILED, None, (), None)
        record = CheckpointRecord(
            record_id=record_id,
            revision=1,
            status=status,
            checkpoint_id=checkpoint_id,
            workspace_path=str(self._workspace),
            shadow_git_dir=str(self._git_dir),
            session_id=session_id,
            boundary_id=boundary_id,
            turn_id=turn_id,
            changed_paths=changed_paths,
            created_at=self._now_fn().isoformat(),
        )
        try:
            self._append_record(record)
        except (OSError, StorageCorruptionError, UnicodeError) as exc:
            logger.debug("checkpoint catalogue write failed: {}", exc)
            return CheckpointResult(CheckpointStatus.FAILED, None, (), None)
        return CheckpointResult(status, checkpoint_id, changed_paths, record_id)

    def get_record(self, record_id: str) -> CheckpointRecord | None:
        """Return one durable record without mutating checkpoint or Workspace state."""

        if not isinstance(record_id, str) or not record_id:
            return None
        return self._load_catalogue().get(record_id)

    def list_records(self) -> tuple[CheckpointRecord, ...]:
        """Return the latest immutable revision of every durable record."""

        return tuple(self._load_catalogue().values())

    def correlate(
        self,
        record_id: str,
        *,
        session_id: str | None,
        boundary_id: str | None,
        turn_id: str | None,
    ) -> CheckpointRecord | None:
        """Append correlation facts after the Session boundary is durable."""

        try:
            current = self.get_record(record_id)
            if current is None:
                return None
            values = {
                "session_id": session_id,
                "boundary_id": boundary_id,
                "turn_id": turn_id,
            }
            for field, value in values.items():
                old = getattr(current, field)
                if old is not None and old != value:
                    logger.debug("checkpoint correlation rejected conflicting {}", field)
                    return None
            updated = replace(current, revision=current.revision + 1, **values)
            self._append_record(updated)
            return updated
        except (OSError, StorageCorruptionError, UnicodeError) as exc:
            logger.debug("checkpoint correlation write failed: {}", exc)
            return None

    async def _git(self, *args: str) -> tuple[int, str, str]:
        """针对 Shadow Repo 运行 Git Command，并返回 ``(rc, out, err)``。

        Command 固定带独立 git-dir/work-tree 与 ``core.quotePath=false``，使 CJK/Japanese/Emoji
        等 non-ASCII Path 保持真实 UTF-8，而不是 Git 默认 Octal Escape；否则 ``edited_files`` 会以
        ``"\\346\\265\\213"`` gibberish 进入 Recovery Prompt。Subprocess cwd 是真实 Workspace。

        communicate 超过 `_GIT_TIMEOUT_SECONDS` 会 Kill/Wait Process，并合成 ``(-1, "", "timeout")``；
        正常输出用 replacement decoding，调用方据 rc 决定 best-effort degradation。
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
        """Lazy Initialize Shadow Repo，幂等返回是否 Ready。

        HEAD 不存在时创建 Parent 并 ``git init``，同时 best-effort 写 NOTICE.txt 解释目录用途、删除
        与禁用方式。每次 Ready 前都会刷新 ``info/exclude``，并配置 gc.auto=256、
        gc.autoDetach=false；前台 GC 避免与 Test/Shutdown rmtree 竞争，且受统一 Git timeout。

        成功设置 `_ready=True`；Init/Git/OSError 记录 Debug 并返回 ``False``，不让 Checkpoint
        Configuration 或 Disk Failure 破坏主 Turn。
        """
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

    async def commit_turn(
        self,
        label: str,
        *,
        session_id: str | None = None,
        boundary_id: str | None = None,
        turn_id: str | None = None,
    ) -> CheckpointResult:
        """把当前 Worktree Snapshot 为一个带 ``label`` 的 Shadow Commit。

        Ready 后依次执行 ``add -A``、cached name-only diff、Commit、full HEAD。Changed Files 必须
        在 Commit 前捕获，才能精确表示本 Turn staged state；无变化不创建空 Commit。成功返回
        ``(checkpoint_id, changed_files)``，增加实例 Commit count 并触发 `_maybe_gc`。

        Result 明确区分 CREATED、UNCHANGED 与 FAILED；仍支持旧调用方的二值解包。完整 Commit id
        与关联事实写入 Catalogue。任何失败都 best-effort 返回 FAILED，不中断 Agent Turn。
        """
        if not await self._ensure_init():
            return self._record_result(
                CheckpointStatus.FAILED,
                session_id=session_id,
                boundary_id=boundary_id,
                turn_id=turn_id,
            )
        try:
            rc, _, err = await self._git("add", "-A")
            if rc != 0:
                logger.debug("checkpoint add failed: {}", err.strip())
                return self._record_result(
                    CheckpointStatus.FAILED,
                    session_id=session_id,
                    boundary_id=boundary_id,
                    turn_id=turn_id,
                )
            # 当前 Turn 暂存的文件就是本 Turn 的变更，需在提交前捕获。
            rc, out, err = await self._git("diff", "--cached", "--name-only")
            if rc != 0:
                logger.debug("checkpoint diff failed: {}", err.strip())
                return self._record_result(
                    CheckpointStatus.FAILED,
                    session_id=session_id,
                    boundary_id=boundary_id,
                    turn_id=turn_id,
                )
            changed = [ln for ln in out.splitlines() if ln.strip()]
            if not changed:
                return self._record_result(
                    CheckpointStatus.UNCHANGED,
                    session_id=session_id,
                    boundary_id=boundary_id,
                    turn_id=turn_id,
                )
            rc, _, err = await self._git(*_GIT_IDENT, "commit", "-m", label)
            if rc != 0:
                logger.debug("checkpoint commit failed: {}", err.strip())
                return self._record_result(
                    CheckpointStatus.FAILED,
                    session_id=session_id,
                    boundary_id=boundary_id,
                    turn_id=turn_id,
                )
            rc, out, err = await self._git("rev-parse", "HEAD")
            cid = out.strip() if rc == 0 else ""
            if len(cid) not in {40, 64} or any(char not in "0123456789abcdef" for char in cid):
                logger.debug("checkpoint rev-parse failed: {}", err.strip())
                return self._record_result(
                    CheckpointStatus.FAILED,
                    session_id=session_id,
                    boundary_id=boundary_id,
                    turn_id=turn_id,
                )
            self._commit_count += 1
            await self._maybe_gc()
            return self._record_result(
                CheckpointStatus.CREATED,
                checkpoint_id=cid,
                changed_paths=tuple(changed),
                session_id=session_id,
                boundary_id=boundary_id,
                turn_id=turn_id,
            )
        except OSError as exc:
            logger.debug("checkpoint commit error: {}", exc)
            return self._record_result(
                CheckpointStatus.FAILED,
                session_id=session_id,
                boundary_id=boundary_id,
                turn_id=turn_id,
            )

    async def validate(
        self,
        record_id: str,
        *,
        expected_workspace: Path | None = None,
        expected_session_id: str | None = None,
        expected_boundary_id: str | None = None,
        expected_turn_id: str | None = None,
    ) -> CheckpointValidation:
        """Validate durable recovery evidence without restoring or staging files."""

        try:
            record = self.get_record(record_id)
        except (OSError, StorageCorruptionError, UnicodeError):
            return CheckpointValidation(False, "catalogue_corrupt")
        if record is None:
            return CheckpointValidation(False, "record_missing")
        expected_workspace_path = (
            str(Path(expected_workspace).expanduser().resolve())
            if expected_workspace is not None
            else str(self._workspace)
        )
        if record.workspace_path != expected_workspace_path or record.workspace_path != str(self._workspace):
            return CheckpointValidation(False, "workspace_mismatch", record=record)
        if record.shadow_git_dir != str(self._git_dir):
            return CheckpointValidation(False, "shadow_repository_mismatch", record=record)
        correlations = (
            ("session_mismatch", expected_session_id, record.session_id),
            ("boundary_mismatch", expected_boundary_id, record.boundary_id),
            ("turn_mismatch", expected_turn_id, record.turn_id),
        )
        for reason, expected, actual in correlations:
            if expected is not None and expected != actual:
                return CheckpointValidation(False, reason, record=record)
        if record.status is CheckpointStatus.UNCHANGED:
            return CheckpointValidation(False, "checkpoint_unchanged", record=record)
        if record.status is CheckpointStatus.FAILED:
            return CheckpointValidation(False, "checkpoint_failed", record=record)
        if not self._git_dir.is_dir():
            return CheckpointValidation(False, "shadow_repository_missing", record=record)
        checkpoint_id = record.checkpoint_id
        if checkpoint_id is None:
            return CheckpointValidation(False, "checkpoint_object_missing", record=record)
        rc, _, _ = await self._git("cat-file", "-e", f"{checkpoint_id}^{{commit}}")
        if rc != 0:
            return CheckpointValidation(False, "checkpoint_object_missing", record=record)
        rc, tree_out, _ = await self._git("rev-parse", f"{checkpoint_id}^{{tree}}")
        tree_id = tree_out.strip()
        if rc != 0 or len(tree_id) not in {40, 64}:
            return CheckpointValidation(False, "checkpoint_tree_invalid", record=record)
        rc, _, _ = await self._git("cat-file", "-e", f"{tree_id}^{{tree}}")
        if rc != 0:
            return CheckpointValidation(False, "checkpoint_tree_invalid", record=record)
        rc, diff_out, _ = await self._git("diff", "--name-only", checkpoint_id, "--")
        if rc != 0:
            return CheckpointValidation(False, "workspace_drift_unknown", record=record, tree_id=tree_id)
        rc, untracked_out, _ = await self._git("ls-files", "--others", "--exclude-standard")
        if rc != 0:
            return CheckpointValidation(False, "workspace_drift_unknown", record=record, tree_id=tree_id)
        drifted_paths = tuple(
            dict.fromkeys(
                path
                for path in (*diff_out.splitlines(), *untracked_out.splitlines())
                if path.strip()
            )
        )
        return CheckpointValidation(
            True,
            None,
            record=record,
            tree_id=tree_id,
            workspace_drifted=bool(drifted_paths),
            drifted_paths=drifted_paths,
        )

    async def export_archive(self, checkpoint_id: str, destination: Path) -> tuple[bool, str]:
        """Write one validated Git tree as a tar archive without touching either work-tree."""

        rc, _, err = await self._git(
            "archive",
            "--format=tar",
            "-o",
            str(Path(destination).resolve()),
            checkpoint_id,
        )
        return rc == 0, err.strip()

    async def _maybe_gc(self) -> None:
        """周期运行 ``git gc --auto``，避免 Long-lived Session 无限累积 Loose Objects。

        ``--auto`` 在低于 ``gc.auto``（默认 256 Loose Objects）时是 no-op，steady state 成本只是
        Cheap rev-list count，不会每 50 次都 Repack。``_commit_count`` 是 per-instance，重建
        CheckpointService（例如 Fresh AgentLoop Start）会重置，所以 50-commit heartbeat 只是 Hint。

        Load-bearing safety net 是 Init 时设置且跨 Process Restart 仍由 Repo 保存的 ``gc.auto=256``。
        Heartbeat 禁用或 GC failure 都不影响 Turn，失败只记录 Debug。
        """
        if _GC_EVERY_N_COMMITS <= 0:
            return
        if self._commit_count % _GC_EVERY_N_COMMITS != 0:
            return
        rc, _, err = await self._git("gc", "--auto")
        if rc != 0:
            logger.debug("checkpoint gc failed: {}", err.strip())


__all__ = [
    "CheckpointCatalogueError",
    "CheckpointRecord",
    "CheckpointResult",
    "CheckpointService",
    "CheckpointStatus",
    "CheckpointValidation",
]
