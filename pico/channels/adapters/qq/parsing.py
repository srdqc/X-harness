"""Pure routing/content helpers for the QQ adapter.

Resolve the chat/user route and chat-type from a botpy message, and normalize
its text. No I/O — unit-tested directly. The botpy SDK orchestration lives in
:mod:`.channel`.
"""

from __future__ import annotations

from typing import Any

from pico.channels.media import safe_name

_ATTACHMENT_KINDS = (("image/", "image"), ("video/", "video"), ("audio/", "voice"))


def clean_content(data: Any) -> str:
    return (getattr(data, "content", None) or "").strip()


def attachment_labels(data: Any) -> list[str]:
    """Text labels for a botpy message's inbound attachments.

    botpy exposes attachment metadata (``content_type`` / ``filename``) on
    ``BaseMessage``, so labelling is deterministic and needs no I/O. The bytes
    themselves sit behind an ephemeral platform URL that botpy offers no
    download helper for, so only the label reaches the Turn — QQ inbound media
    is never fetched or persisted (unlike Feishu and WeCom, whose SDKs expose a
    download call). Server-supplied filenames pass through ``safe_name`` so a
    crafted name cannot smuggle path components into the Turn text.
    """
    labels: list[str] = []
    for item in getattr(data, "attachments", None) or []:
        content_type = (getattr(item, "content_type", None) or "").lower()
        kind = next((k for prefix, k in _ATTACHMENT_KINDS if content_type.startswith(prefix)), "file")
        filename = getattr(item, "filename", None)
        labels.append(f"[{kind}: {safe_name(filename)}]" if filename else f"[{kind}]")
    return labels


def compose_content(data: Any) -> str:
    """Normalized inbound text: the message body plus one line per attachment."""
    parts = [text] if (text := clean_content(data)) else []
    parts.extend(attachment_labels(data))
    return "\n".join(parts)


def resolve_route(data: Any, is_group: bool) -> tuple[str, str, str]:
    """Return ``(chat_id, user_id, chat_type)`` for an inbound message.

    Group messages route by group openid (sender = member openid). Guild
    direct messages (botpy ``DirectMessage``) carry a ``guild_id`` — the DM
    session id that replies must go back through (``post_dms``), distinct from
    QQ C2C. Plain C2C messages route by the author's id, falling back to
    user_openid.
    """
    if is_group:
        return data.group_openid, data.author.member_openid, "group"
    user_id = str(getattr(data.author, "id", None) or getattr(data.author, "user_openid", "unknown"))
    if guild_id := getattr(data, "guild_id", None):
        return str(guild_id), user_id, "guild_dm"
    return user_id, user_id, "c2c"
