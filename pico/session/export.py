"""Render and export a stored Session.

Pure rendering (``render_transcript``) is separated from the file write
(``write_transcript``). Public export surfaces write a canonical JSON envelope
that preserves the complete Session payload and carries a SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pico.session.manager import Session
from pico.utils.helpers import ensure_dir, safe_filename

_ROLE_HEADINGS = {
    "user": "## 🧑 User",
    "assistant": "## 🤖 Assistant",
    "system": "## ⚙️ System",
    "tool": "## 🛠 Tool result",
}
_EXPORT_SCHEMA = "pico.session.export.v1"


def render_transcript(session: Session) -> str:
    """Render ``session`` to a full-fidelity Markdown transcript.

    Includes a header (key, timestamps, message count, title when set) and each
    message in order: user/assistant/system/tool under distinct headings, the
    assistant reasoning block when present, and tool calls/results as fenced
    blocks. Pure — performs no I/O.
    """
    parts: list[str] = [_render_header(session)]
    for msg in session.messages:
        parts.append(_render_message(msg))
    return "\n\n".join(p for p in parts if p) + "\n"


def default_export_path(workspace: Path, key: str) -> Path:
    """Default destination for a portable Session export."""

    return Path(workspace) / "exports" / f"{safe_filename(key)}.pico-session.json"


def write_transcript(session: Session, dest: Path) -> Path:
    """Render ``session`` and write it to ``dest``, returning the absolute path.

    Creates the parent directory if absent and overwrites any existing file so
    a re-export reflects the session's current state.
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    dest.write_text(render_transcript(session), encoding="utf-8")
    return dest.resolve()


def build_portable_export(session: Session) -> dict[str, Any]:
    """Build a complete, self-verifying Session export envelope."""
    payload = {
        "key": session.key,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "metadata": session.metadata,
        "last_consolidated": session.last_consolidated,
        "pending_clarification": session.pending_clarification,
        "messages": session.messages,
        "message_count": len(session.messages),
        "transcript_markdown": render_transcript(session),
    }
    return {
        "schema": _EXPORT_SCHEMA,
        "payload": payload,
        "sha256": _payload_digest(payload),
    }


def write_portable_export(session: Session, dest: Path) -> Path:
    """Write a canonical portable export and return its absolute path."""
    dest = Path(dest)
    ensure_dir(dest.parent)
    envelope = build_portable_export(session)
    dest.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dest.resolve()


def verify_export(path: Path) -> bool:
    """Return whether ``path`` is a structurally valid, untampered export."""
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(envelope, dict) or envelope.get("schema") != _EXPORT_SCHEMA:
        return False
    payload = envelope.get("payload")
    digest = envelope.get("sha256")
    if not isinstance(payload, dict) or not isinstance(digest, str):
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list) or payload.get("message_count") != len(messages):
        return False
    required = {
        "key",
        "created_at",
        "updated_at",
        "metadata",
        "last_consolidated",
        "pending_clarification",
        "messages",
        "message_count",
        "transcript_markdown",
    }
    return required <= payload.keys() and digest == _payload_digest(payload)


def _payload_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ── 内部实现 ─────────────────────────────────────────────────────────


def _render_header(session: Session) -> str:
    title = (session.metadata or {}).get("title")
    meta = (
        f"_{session.created_at.isoformat(timespec='seconds')}"
        f" → {session.updated_at.isoformat(timespec='seconds')}"
        f" · {len(session.messages)} messages_"
    )
    lines = [f"# Session `{session.key}`", meta]
    if title:
        lines.insert(1, f"**{title}**")
    return "\n".join(lines)


def _render_message(msg: dict[str, Any]) -> str:
    role = msg.get("role", "")
    heading = _ROLE_HEADINGS.get(role, f"## {role or 'message'}")
    if role == "tool":
        name = msg.get("name") or msg.get("tool_call_id") or ""
        suffix = f": `{name}`" if name else ""
        return f"{heading}{suffix}\n\n{_fenced(_as_text(msg.get('content')))}"

    blocks: list[str] = [heading]
    reasoning = _reasoning_text(msg)
    if reasoning:
        quoted = "\n".join(f"> {line}" for line in reasoning.splitlines() or [""])
        blocks.append(f"> 💭 _thinking_\n{quoted}")
    content = _as_text(msg.get("content"))
    if content:
        blocks.append(content)
    for call in msg.get("tool_calls") or []:
        blocks.append(_render_tool_call(call))
    return "\n\n".join(blocks)


def _render_tool_call(call: dict[str, Any]) -> str:
    fn = call.get("function") or {}
    name = fn.get("name") or call.get("name") or "tool"
    args = fn.get("arguments")
    if args is None:
        args = call.get("arguments")
    return f"⏺ **{name}**\n\n{_fenced(_as_text(args))}"


def _reasoning_text(msg: dict[str, Any]) -> str:
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return rc
    blocks = msg.get("thinking_blocks")
    if isinstance(blocks, list):
        texts = [b.get("thinking", "") for b in blocks if isinstance(b, dict) and b.get("thinking")]
        if texts:
            return "\n".join(texts)
    return ""


def _as_text(content: Any) -> str:
    """Flatten a message content value (str, multimodal list, or dict) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    out.append(part["text"])
                elif part.get("type") == "image_url":
                    out.append("[image]")
                else:
                    out.append(json.dumps(part, ensure_ascii=False))
            else:
                out.append(str(part))
        return "\n".join(out)
    return json.dumps(content, ensure_ascii=False)


def _fenced(text: str) -> str:
    return f"```\n{text}\n```"


__all__ = [
    "build_portable_export",
    "default_export_path",
    "render_transcript",
    "verify_export",
    "write_portable_export",
    "write_transcript",
]
