"""Session management for conversation history."""

import copy
import errno
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from pico.utils.atomic_io import (
    StorageCorruptionError,
    atomic_replace,
    epoch_is_known,
    locked_append,
    locked_delete,
    locked_read,
    read_epoch,
    read_utf8_with_incomplete_tail,
)
from pico.utils.helpers import ensure_dir, safe_filename


def new_chat_id(now: datetime | None = None) -> str:
    """Mint an opaque, sortable per-session chat_id: ``YYYYMMDD_HHMMSS_xxxxxx``.

    Sortable by value (timestamp prefix) and collision-safe (uuid suffix);
    channel-agnostic. Becomes the session key's chat_id segment and the JSONL
    filename stem.
    """
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class SessionResolution:
    """Outcome of resolving a user-supplied session id to a full key.

    ``status`` is one of ``"resolved"`` / ``"ambiguous"`` / ``"not_found"``.
    ``key`` carries the full ``channel:chat_id`` when resolved; ``candidates``
    carries the matching full keys when ambiguous. The no-match case is reported
    as ``not_found`` so each caller decides its own tail — the agent
    ``--session`` path mints ``cli:<value>``, while a read-only export errors.
    """

    status: str
    key: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id 格式的键
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # 已归并到文件的消息数量
    # ── 个性化状态 ───────────────────────────────────────────────────────────
    # Agent 提出澄清问题并等待回答时设置。
    # 结构：{"original_message": str, "question": str, "domain": str}
    # 用户回答处理完毕后立即清除。
    pending_clarification: dict | None = field(default=None)
    # 已落盘的消息数量；save() 只追加该索引之后的消息。
    _persisted_count: int = field(default=0, repr=False)
    # 用于区分已保存的空会话和从未持久化的惰性会话。
    _persisted: bool = field(default=False, repr=False, compare=False)
    _storage_epoch: int = field(default=0, repr=False, compare=False)
    _persisted_snapshot: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _requires_rewrite: bool = field(default=False, repr=False, compare=False)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        self.record({"role": role, "content": content, **kwargs})

    def record(self, msg: dict[str, Any]) -> None:
        """Append a message dict, stamping a wall-clock timestamp.

        The single choke point for session writes — every persistence path
        (``add_message``, the agent loop's ``_save_turn``, clarification
        appends) must come through here so no message lands unstamped. A
        caller-set ``timestamp`` is preserved. Per-message ordering and
        turn grouping derive from append order and the ``role`` boundary,
        so no separate received_at / turn_id stamp is kept.
        """
        msg.setdefault("timestamp", datetime.now().isoformat())
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a user turn."""
        unconsolidated = self.messages[self.last_consolidated :]
        sliced = unconsolidated[-max_messages:]

        # 丢弃开头的非用户消息，避免产生孤立的 tool_result 块。
        for i, m in enumerate(sliced):
            if m.get("role") == "user":
                sliced = sliced[i:]
                break

        out: list[dict[str, Any]] = []
        for m in sliced:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.pending_clarification = None
        self.updated_at = datetime.now()

    def undo_last_turn(self, n: int = 1) -> int:
        """Drop the last ``n`` user-turn blocks from the unconsolidated tail.

        A turn starts at a ``role == "user"`` message and runs to the next
        user message (its assistant/tool followers inherit it). Only the
        unconsolidated tail (``messages[last_consolidated:]``) is eligible —
        content already summarized into MEMORY.md is never crossed. Returns
        the number of messages removed (0 when the tail has no user message).
        Persistence is the caller's job via ``SessionManager.save``.
        """
        if n < 1:
            return 0
        start = self.last_consolidated
        user_starts = [i for i in range(start, len(self.messages)) if self.messages[i].get("role") == "user"]
        if not user_starts:
            return 0
        cut_index = user_starts[-n] if n <= len(user_starts) else user_starts[0]
        removed = len(self.messages) - cut_index
        self.messages = self.messages[:cut_index]
        self.pending_clarification = None
        self.updated_at = datetime.now()
        return removed

    def _persistence_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "messages": self.messages,
                "metadata": self.metadata,
                "last_consolidated": self.last_consolidated,
                "pending_clarification": self.pending_clarification,
            }
        )

    def _mark_persisted(self) -> None:
        self._persisted_snapshot = self._persistence_snapshot()

    def _is_dirty(self) -> bool:
        if self._requires_rewrite:
            return True
        if not self._persisted:
            return bool(
                self.messages or self.metadata or self.last_consolidated or self.pending_clarification is not None
            )
        return self._persisted_snapshot != self._persistence_snapshot()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session: sessions/{channel}/{chat_id}.jsonl."""
        channel, _, chat_id = key.partition(":")
        return self.sessions_dir / safe_filename(channel) / f"{safe_filename(chat_id)}.jsonl"

    @staticmethod
    def key_from_path(path: Path) -> str:
        """Best-effort reverse of the nested filename encoding for a session
        file: channel is the parent directory, chat_id is the stem.

        The on-disk ``_type:metadata`` key is authoritative when present and
        wins over this; callers use it only as the fallback for metadata-less
        files. ``safe_filename`` is non-invertible, so any character it folds
        to ``_`` (``/``, ``:``, ...) is not recovered here.
        """
        return f"{path.parent.name}:{path.stem}"

    @staticmethod
    def _validate_metadata_identity(record_key: Any, requested_key: str) -> bool:
        if not isinstance(record_key, str):
            raise StorageCorruptionError(f"session {requested_key} metadata key must be a string")
        if record_key != requested_key:
            raise StorageCorruptionError(
                f"session metadata key {record_key!r} does not match requested key {requested_key!r}"
            )
        return True

    @classmethod
    def _validate_fallback_identity(
        cls,
        path: Path,
        requested_key: str,
        *,
        has_metadata_key: bool,
    ) -> None:
        if has_metadata_key:
            return
        canonical_key = cls.key_from_path(path)
        if requested_key != canonical_key:
            raise StorageCorruptionError(
                f"requested key {requested_key!r} does not match canonical path key {canonical_key!r}"
            )

    @classmethod
    def _decode_payload(cls, path: Path, requested_key: str, raw: str) -> dict[str, Any]:
        records = [
            (line_number, line.strip()) for line_number, line in enumerate(raw.splitlines(), start=1) if line.strip()
        ]
        messages: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}
        created_at: datetime | None = None
        updated_at: datetime | None = None
        last_consolidated = 0
        pending_clarification: dict[str, Any] | None = None
        has_metadata_key = False
        partial_tail_found = False
        for index, (line_number, line) in enumerate(records):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                partial_tail = index == len(records) - 1 and not raw.endswith(("\n", "\r"))
                if partial_tail:
                    logger.debug("Skipping partial trailing line in session {}", requested_key)
                    partial_tail_found = True
                    continue
                raise StorageCorruptionError(
                    f"session {requested_key} has invalid JSON record at line {line_number}"
                ) from exc
            if not isinstance(data, dict):
                raise StorageCorruptionError(f"session {requested_key} record must be an object (line {line_number})")
            if data.get("_type") == "metadata":
                record_key = data.get("key")
                record_metadata = data.get("metadata", {})
                record_created_at = data.get("created_at")
                record_updated_at = data.get("updated_at")
                record_last_consolidated = data.get("last_consolidated", 0)
                record_pending_clarification = data.get("pending_clarification")
                has_metadata_key = cls._validate_metadata_identity(record_key, requested_key) or has_metadata_key
                if not isinstance(record_metadata, dict):
                    raise StorageCorruptionError(f"session {requested_key} metadata payload must be an object")
                if record_created_at is not None and not isinstance(record_created_at, str):
                    raise StorageCorruptionError(f"session {requested_key} created_at must be a string")
                if record_updated_at is not None and not isinstance(record_updated_at, str):
                    raise StorageCorruptionError(f"session {requested_key} updated_at must be a string")
                if (
                    isinstance(record_last_consolidated, bool)
                    or not isinstance(record_last_consolidated, int)
                    or record_last_consolidated < 0
                ):
                    raise StorageCorruptionError(
                        f"session {requested_key} last_consolidated must be a non-negative integer"
                    )
                if record_pending_clarification is not None and not isinstance(record_pending_clarification, dict):
                    raise StorageCorruptionError(f"session {requested_key} pending_clarification must be an object")
                metadata = record_metadata
                try:
                    created_at = datetime.fromisoformat(record_created_at) if record_created_at else None
                    updated_at = datetime.fromisoformat(record_updated_at) if record_updated_at else None
                except ValueError as exc:
                    raise StorageCorruptionError(f"session {requested_key} timestamp is invalid") from exc
                last_consolidated = record_last_consolidated
                pending_clarification = record_pending_clarification
            else:
                messages.append(data)
        cls._validate_fallback_identity(
            path,
            requested_key,
            has_metadata_key=has_metadata_key,
        )
        return {
            "messages": messages,
            "metadata": metadata,
            "created_at": created_at,
            "updated_at": updated_at,
            "last_consolidated": last_consolidated,
            "pending_clarification": pending_clarification,
            "partial_tail_found": partial_tail_found,
        }

    @staticmethod
    def _durable_state(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": payload["messages"],
            "metadata": payload["metadata"],
            "last_consolidated": payload["last_consolidated"],
            "pending_clarification": payload["pending_clarification"],
        }

    @classmethod
    def _validate_append_base(
        cls,
        path: Path,
        requested_key: str,
        raw: str,
        expected: dict[str, Any],
    ) -> None:
        payload = cls._decode_payload(path, requested_key, raw)
        if payload["partial_tail_found"]:
            raise FileNotFoundError(f"session has a partial trailing record before append: {path}")
        actual = cls._durable_state(payload)
        for state_field in ("metadata", "last_consolidated", "pending_clarification"):
            if actual[state_field] != expected[state_field]:
                raise FileNotFoundError(f"session metadata changed before append: {path}")

    @classmethod
    def _validate_rewrite_base(
        cls,
        path: Path,
        requested_key: str,
        raw: str,
        expected: dict[str, Any],
    ) -> None:
        actual = cls._durable_state(cls._decode_payload(path, requested_key, raw))
        if actual != expected:
            raise FileNotFoundError(f"session changed before rewrite: {path}")

    def resolve_key(self, value: str) -> SessionResolution:
        """Resolve a session id to a full ``channel:chat_id`` key across channels.

        Shared resolution core for the agent ``--session`` path and session
        export:

        - a value containing ':' is already a full key -> resolved;
        - exactly one exact chat_id match across channels -> resolved;
        - exactly one prefix match -> resolved;
        - more than one match -> ambiguous (candidate full keys);
        - no match -> not_found.

        The no-match tail is reported as ``not_found``; callers decide whether to
        mint (agent ``--session``) or error (read-only export).
        """
        if ":" in value:
            return SessionResolution("resolved", key=value)
        sessions = self.list_sessions(channel=None)
        exact = [s for s in sessions if s["key"].partition(":")[2] == value]
        matches = exact or [s for s in sessions if s["key"].partition(":")[2].startswith(value)]
        if len(matches) > 1:
            return SessionResolution("ambiguous", candidates=tuple(s["key"] for s in matches))
        if matches:
            return SessionResolution("resolved", key=matches[0]["key"])
        return SessionResolution("not_found")

    def find_most_recent_chat_id(self, channel: str) -> str | None:
        """Return the chat_id of the most-recently-updated session on this
        channel, or None if no such session exists.

        Used by cron delivery at trigger time to auto-resolve where to
        forward ephemeral (cli / tui) reminders, so users don't need to
        know their own open_id / chat_id on the target channel.

        Reads each candidate file's metadata line (first line of the JSONL)
        to get the authoritative session key ``<channel>:<chat_id>`` and
        ``updated_at``; recency is decided by ``updated_at``, falling back
        to file mtime for files that lack it.
        """
        channel_dir = self.sessions_dir / safe_filename(channel)
        if not channel_dir.is_dir():
            return None

        best_chat_id: str | None = None
        best_updated = ""
        for p in channel_dir.glob("*.jsonl"):
            meta, _count = self._scan_file(p)
            if meta is None:
                continue
            key_val = meta.get("key", "")
            if ":" not in key_val:
                continue
            ch, chat_id = key_val.split(":", 1)
            if ch != channel or not chat_id:
                continue
            updated = meta.get("updated_at")
            if not isinstance(updated, str) or not updated:
                try:
                    updated = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
                except OSError:
                    continue
            if updated > best_updated:
                best_chat_id = chat_id
                best_updated = updated
        return best_chat_id

    @staticmethod
    def _scan_file(path: Path) -> tuple[dict[str, Any] | None, int]:
        """Validate one session file and return its last metadata and message count.

        One metadata record is appended per save, so the last reflects
        current state. Message lines are counted without keeping them in
        memory.
        """
        meta: dict[str, Any] | None = None
        identity: str | None = None
        count = 0
        try:
            raw = read_utf8_with_incomplete_tail(path)
            records = [line.strip() for line in raw.splitlines() if line.strip()]
            for index, line in enumerate(records):
                line = line.strip()
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(records) - 1 and not raw.endswith(("\n", "\r")):
                        break
                    return None, 0
                if not isinstance(data, dict):
                    return None, 0
                if data.get("_type") == "metadata":
                    key = data.get("key")
                    if not isinstance(key, str):
                        return None, 0
                    channel, separator, chat_id = key.partition(":")
                    if (
                        not separator
                        or path.parent.name != safe_filename(channel)
                        or path.stem != safe_filename(chat_id)
                        or (identity is not None and key != identity)
                    ):
                        return None, 0
                    record_metadata = data.get("metadata", {})
                    created_at = data.get("created_at")
                    updated_at = data.get("updated_at")
                    last_consolidated = data.get("last_consolidated", 0)
                    pending_clarification = data.get("pending_clarification")
                    if not isinstance(record_metadata, dict):
                        return None, 0
                    if created_at is not None and not isinstance(created_at, str):
                        return None, 0
                    if updated_at is not None and not isinstance(updated_at, str):
                        return None, 0
                    if (
                        isinstance(last_consolidated, bool)
                        or not isinstance(last_consolidated, int)
                        or last_consolidated < 0
                    ):
                        return None, 0
                    if pending_clarification is not None and not isinstance(pending_clarification, dict):
                        return None, 0
                    identity = key
                    meta = data
                else:
                    count += 1
        except (OSError, UnicodeError):
            return None, 0
        return meta, count

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session, storage_epoch = self._load_state(key)
        if session is None:
            session = Session(key=key, _storage_epoch=storage_epoch)

        self._cache[key] = session
        return session

    def _load_state(self, key: str) -> tuple[Session | None, int]:
        """Load a session and its missing-path generation from one snapshot."""
        path = self._get_session_path(key)

        try:
            try:
                loaded = locked_read(path)
            except OSError as exc:
                if not isinstance(exc, PermissionError) and exc.errno not in {errno.EACCES, errno.EROFS}:
                    raise
                loaded = None
                for _ in range(3):
                    epoch_before = read_epoch(path)
                    known_before = epoch_is_known(path)
                    try:
                        raw = read_utf8_with_incomplete_tail(path)
                    except FileNotFoundError:
                        raw = None
                    epoch_after = read_epoch(path)
                    known_after = epoch_is_known(path)
                    exists_after = path.exists()
                    if (
                        epoch_before == epoch_after
                        and known_before == known_after
                        and exists_after == (raw is not None)
                    ):
                        loaded = (raw, epoch_after, known_after)
                        break
            if loaded is None:
                raise StorageCorruptionError(f"failed to read a stable snapshot for session {key}")
            raw, storage_epoch, known_epoch = loaded
            if raw is None:
                if known_epoch and storage_epoch == 0:
                    raise StorageCorruptionError(f"session {key} is missing its primary file for a known generation")
                return None, storage_epoch
            payload = self._decode_payload(path, key, raw)

            session = Session(
                key=key,
                messages=payload["messages"],
                created_at=payload["created_at"] or datetime.now(),
                updated_at=payload["updated_at"] or payload["created_at"] or datetime.now(),
                metadata=payload["metadata"],
                last_consolidated=payload["last_consolidated"],
                pending_clarification=payload["pending_clarification"],
                _storage_epoch=storage_epoch,
                _requires_rewrite=payload["partial_tail_found"],
            )
            session._persisted_count = len(session.messages)
            session._persisted = True
            session._mark_persisted()
            return session, storage_epoch
        except StorageCorruptionError:
            raise
        except Exception as exc:
            logger.warning("Failed to load session {}: {}", key, exc)
            raise StorageCorruptionError(f"failed to load session {key}") from exc

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        session, _storage_epoch = self._load_state(key)
        return session

    def save(self, session: Session, *, force_rewrite: bool = False) -> None:
        """Save a session to disk.

        Appends a fresh metadata record plus the not-yet-persisted messages
        under a cross-process lock, so concurrent writers never lose each
        other's turns and a turn's messages stay contiguous. A shrunken
        message list or forced history fence rewrites the file atomically.
        """
        path = self._get_session_path(session.key)

        channel, _, chat_id = session.key.partition(":")
        reserved = {
            "source": None,
            "channel": channel,
            "chat_id": chat_id,
            "title": None,
            "parent_session_id": None,
        }
        session.metadata = {**reserved, **session.metadata}

        metadata_line = json.dumps(
            {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
                # 个性化状态：跨重启保留澄清问题的等待状态。
                "pending_clarification": session.pending_clarification,
            },
            ensure_ascii=False,
        )

        persisted_messages = (
            session._persisted_snapshot.get("messages") if session._persisted_snapshot is not None else None
        )
        append_only = (
            persisted_messages is not None
            and len(session.messages) >= len(persisted_messages)
            and session.messages[: len(persisted_messages)] == persisted_messages
        )

        rewrite = force_rewrite or session._requires_rewrite or (session._persisted and not append_only)
        if rewrite:
            lines = [metadata_line]
            lines += [json.dumps(m, ensure_ascii=False) for m in session.messages]
            expected_state = session._persisted_snapshot
            storage_epoch = atomic_replace(
                path,
                "".join(line + "\n" for line in lines),
                expected_epoch=session._storage_epoch,
                expected_exists=session._persisted,
                require_existing=session._persisted,
                increment_epoch=True,
                validate_existing=(
                    None
                    if expected_state is None
                    else lambda raw: self._validate_rewrite_base(
                        path,
                        session.key,
                        raw,
                        expected_state,
                    )
                ),
            )
        else:
            new_messages = session.messages[session._persisted_count :]
            lines = [metadata_line]
            lines += [json.dumps(m, ensure_ascii=False) for m in new_messages]
            expected_state = (
                session._persisted_snapshot
                if session._persisted_snapshot is not None
                else session._persistence_snapshot()
            )
            storage_epoch = locked_append(
                path,
                lines,
                expected_epoch=session._storage_epoch,
                require_existing=session._persisted,
                validate_existing=lambda raw: self._validate_append_base(
                    path,
                    session.key,
                    raw,
                    expected_state,
                ),
            )

        session._persisted_count = len(session.messages)
        session._persisted = True
        session._storage_epoch = storage_epoch
        session._requires_rewrite = False
        session._mark_persisted()
        self._cache[session.key] = session

    def commit_history_rewrite(self, session: Session) -> None:
        """Commit a clear or undo while fencing writers of the old history."""
        if session._persisted or session._is_dirty():
            self.save(session, force_rewrite=True)
            return

        path = self._get_session_path(session.key)
        locked_delete(
            path,
            expected_epoch=session._storage_epoch,
            expected_exists=False,
            fence_missing=True,
            increment_epoch=True,
        )
        session._storage_epoch += 1
        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def delete(
        self,
        key: str,
        *,
        allow_cached_missing: bool = False,
        expected_epoch: int | None = None,
        expected_exists: bool | None = None,
    ) -> bool:
        """Invalidate the current logical session generation.

        Returns True when the matching persisted or cached lazy Session was
        removed. ``allow_cached_missing`` treats a known lazy Session as
        logically deleted after fencing stale references. Deleting an unknown
        key is a safe no-op.
        """
        path = self._get_session_path(key)
        cached = key in self._cache
        try:
            removed = locked_delete(
                path,
                expected_epoch=expected_epoch,
                expected_exists=expected_exists,
                fence_missing=cached,
                increment_epoch=True,
            )
        except OSError:
            logger.warning("session.delete: failed to remove file for {}", key)
            return False
        self.invalidate(key)
        return removed or (allow_cached_missing and cached and expected_exists is False)

    def exists(self, key: str) -> bool:
        """Return True if the session has a file on disk (lazy sessions don't)."""
        return self._get_session_path(key).exists()

    def peek(self, key: str) -> "Session | None":
        """Return the cached session if present; else load from disk without caching.

        Callers that need read-only access to a session should use this instead
        of get_or_create, which would cache a fresh empty session for unknown keys.
        """
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        return self._load(key)

    def fork(self, source_key: str, *, title: str | None = None) -> "Session | None":
        """Fork ``source_key`` at its head into a new diverging child session.

        Full-copy semantics: the child is minted with a fresh chat_id on the
        source's channel, a deep copy of the source's messages, and
        ``parent_session_id`` set to the source key (the reserved lineage slot).
        The child inherits ``last_consolidated`` (so its active-context window
        matches the source at the fork point) and resets ``pending_clarification``
        (interaction wait-state is not history). The child is persisted eagerly.

        Returns the persisted child, or None when the source does not exist or
        has zero messages, or when the source cannot be flushed first.
        """
        source = self.peek(source_key)
        if source is None:
            return None
        if not self.flush(source_key):
            return None
        if not source.messages:
            return None

        channel = source_key.partition(":")[0]
        child = Session(
            key=f"{channel}:{new_chat_id()}",
            messages=copy.deepcopy(source.messages),
            last_consolidated=source.last_consolidated,
        )
        if title is not None:
            child.metadata["title"] = title
        else:
            parent_title = (source.metadata or {}).get("title")
            if parent_title:
                child.metadata["title"] = f"{parent_title} (fork)"
        child.metadata["parent_session_id"] = source_key
        self.save(child)
        return child

    def flush(self, key: str) -> bool:
        """Save the cached session iff it has unpersisted messages.

        Returns False only when a save was attempted and failed (the failure
        is swallowed); True otherwise, including the no-op cases (key not
        cached / no persisted-state changes).
        """
        cached = self._cache.get(key)
        if cached is None:
            return True
        if cached._is_dirty():
            try:
                self.save(cached)
            except Exception:
                logger.warning("flush: failed to persist session {}", key)
                return False
        return True

    def list_sessions(self, channel: str | None = None) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by channel.

        Each entry carries: key, created_at, updated_at, path, message_count.
        Sorted by updated_at descending. Each file is read in a single pass.
        """
        sessions = []

        for path in self.sessions_dir.glob("*/*.jsonl"):
            if channel is not None and path.parent.name != channel:
                continue
            data, message_count = self._scan_file(path)
            if data is None:
                continue
            key = data.get("key") or self.key_from_path(path)
            sessions.append(
                {
                    "key": key,
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "path": str(path),
                    "message_count": message_count,
                    "metadata": data.get("metadata", {}),
                }
            )

        return sorted(
            sessions,
            key=lambda item: item["updated_at"] if isinstance(item.get("updated_at"), str) else "",
            reverse=True,
        )
