"""Sanitize ephemeral multimodal data before writing durable artifacts."""

from __future__ import annotations

import re
from typing import Any

IMAGE_DATA_OMITTED = "[image data omitted]"

_IMAGE_DATA_URI = re.compile(
    r"data:image/[a-z0-9.+-]+"
    r"(?:;[a-z0-9!#$&^_.+-]+=[^;,\s]+)*"
    r";base64,[a-z0-9+/=_-]*",
    re.IGNORECASE,
)


def sanitize_persisted_text(value: str) -> str:
    return _IMAGE_DATA_URI.sub(IMAGE_DATA_OMITTED, value)


def sanitize_persisted_payload(value: Any) -> Any:
    """Return a non-mutating copy with inline image data removed."""
    if isinstance(value, str):
        return sanitize_persisted_text(value)
    if isinstance(value, list):
        return [sanitize_persisted_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_persisted_payload(item) for item in value)
    if isinstance(value, dict):
        return {sanitize_persisted_payload(key): sanitize_persisted_payload(item) for key, item in value.items()}
    if isinstance(value, set):
        return {sanitize_persisted_payload(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(sanitize_persisted_payload(item) for item in value)
    return value


__all__ = [
    "IMAGE_DATA_OMITTED",
    "sanitize_persisted_payload",
    "sanitize_persisted_text",
]
