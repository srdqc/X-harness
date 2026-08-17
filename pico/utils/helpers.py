"""Utility functions for pico."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken


def detect_image_mime(data: bytes) -> str | None:
    """Detect image MIME type from magic bytes, ignoring file extension."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def safe_filename(name: str) -> str:
    """Replace unsafe path characters with underscores."""
    return _UNSAFE_CHARS.sub("_", name).strip()


def split_message(content: str, max_len: int = 2000) -> list[str]:
    """
    Split content into chunks within max_len, preferring line breaks.

    Args:
        content: The text content to split.
        max_len: Maximum length per chunk (default 2000 for Discord compatibility).

    Returns:
        List of message chunks, each within max_len.
    """
    if not content:
        return []
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        # 优先在换行符处截断，其次是空格，最后才强制截断。
        pos = cut.rfind("\n")
        if pos <= 0:
            pos = cut.rfind(" ")
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


def build_assistant_message(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a provider-safe assistant message with optional reasoning fields."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    if thinking_blocks:
        msg["thinking_blocks"] = thinking_blocks
    return msg


def _message_token_parts(message: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))

    for key in ("name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(reasoning)
    if message.get("thinking_blocks"):
        parts.append(json.dumps(message["thinking_blocks"], ensure_ascii=False))
    return parts


def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Estimate prompt tokens with tiktoken."""
    parts: list[str] = []
    for msg in messages:
        parts.extend(_message_token_parts(msg))

    if tools:
        parts.append(json.dumps(tools, ensure_ascii=False))

    payload = "\n".join(parts)
    if not payload:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(payload)))
    except Exception:
        return max(1, len(payload) // 4)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate prompt tokens contributed by one persisted message."""
    payload = "\n".join(_message_token_parts(message))
    if not payload:
        return 1
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(payload)))
    except Exception:
        return max(1, len(payload) // 4)


def estimate_prompt_tokens_chain(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    """Estimate prompt tokens via provider counter first, then tiktoken fallback."""
    provider_counter = getattr(provider, "estimate_prompt_tokens", None)
    if callable(provider_counter):
        try:
            tokens, source = provider_counter(messages, tools, model)
            if isinstance(tokens, (int, float)) and tokens > 0:
                return int(tokens), str(source or "provider_counter")
        except Exception:
            pass

    estimated = estimate_prompt_tokens(messages, tools)
    if estimated > 0:
        return estimated, "tiktoken"
    return 0, "none"


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """Sync bundled templates to workspace. Only creates missing files."""
    from importlib.resources import files as pkg_files

    try:
        tpl = pkg_files("pico") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []

    def _write(src, dest: Path):
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8") if src else "", encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    def _migrate(src: Path, dest: Path):
        """One-shot copy of legacy content to the L4 path. No-op when the
        source is missing or the destination already exists — safe to
        re-run on every workspace sync.  Reads as binary then decodes
        with UTF-8 (replace) so legacy files written under a non-UTF-8
        Windows code page still migrate without crashing."""
        if not src.is_file() or dest.exists():
            return
        try:
            text = src.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        added.append(f"{dest.relative_to(workspace)} (migrated from {src.relative_to(workspace)})")

    # 第一步：把旧版工作区文件迁移到 L4 布局。只有旧文件存在且 L4 目标
    # 尚不存在时规则才生效，因此用户直接对 L4 路径所做的修改优先。
    _migrate(workspace / "memory" / "MEMORY.md", workspace / "user_memory" / "profile" / "user.md")
    _migrate(workspace / "memory" / "HISTORY.md", workspace / "user_memory" / "episodic" / "episodes.md")
    _migrate(workspace / "SOUL.md", workspace / "agent_memory" / "profile" / "soul.md")
    _migrate(workspace / "AGENTS.md", workspace / "agent_memory" / "profile" / "agent.md")
    _migrate(workspace / "USER.md", workspace / "user_memory" / "profile" / "user.md")
    # 第二步：其余缺失文件回退到内置模板。先处理 L4 支柱文件；
    # TOOLS.md 仍保留在工作区根目录。
    _write(tpl / "SOUL.md", workspace / "agent_memory" / "profile" / "soul.md")
    _write(tpl / "AGENTS.md", workspace / "agent_memory" / "profile" / "agent.md")
    _write(tpl / "USER.md", workspace / "user_memory" / "profile" / "user.md")
    _write(None, workspace / "user_memory" / "episodic" / "episodes.md")
    # 程序性记忆文件在旧版布局中没有对应来源。
    _write(None, workspace / "agent_memory" / "procedural" / "skills.md")
    _write(None, workspace / "agent_memory" / "procedural" / "case.md")
    _write(tpl / "TOOLS.md", workspace / "TOOLS.md")
    (workspace / "skills").mkdir(exist_ok=True)

    if added and not silent:
        from rich.console import Console

        _c = Console(stderr=True)
        for name in added:
            _c.print(f"  [dim]Created {name}[/dim]")
    return added
