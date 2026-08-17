"""TUI image attachment handling."""

from __future__ import annotations

import asyncio
import mimetypes
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

from PIL import Image

from pico.spine import Media
from pico.tui_rpc.models import ImageAttachParams

if TYPE_CHECKING:
    from pico.tui_rpc.dispatcher import Dispatcher

_pending_images: dict[str, list[Media]] = {}
_attachment_generations: dict[str, int] = {}
_attachment_global_generation = 0
_IMAGE_MAX_BYTES = 32 * 1024 * 1024
_IMAGE_MAX_COUNT = 8
_IMAGE_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_RASTER_MIMES = {
    "BMP": "image/bmp",
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}
_SVG_MIME = "image/svg+xml"


def _attachment_token(session_id: str) -> tuple[int, int]:
    return _attachment_global_generation, _attachment_generations.get(session_id, 0)


def _is_decodable_raster(data: bytes, expected_mime: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if _RASTER_MIMES.get(image.format or "") != expected_mime:
                    return False
                image.verify()
            with Image.open(BytesIO(data)) as image:
                if _RASTER_MIMES.get(image.format or "") != expected_mime:
                    return False
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, OSError, SyntaxError, ValueError):
        return False

    return True


def _is_decodable_svg(data: bytes) -> bool:
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        return False
    try:
        root = ElementTree.fromstring(data)
    except (ElementTree.ParseError, UnicodeDecodeError, ValueError):
        return False
    return isinstance(root.tag, str) and root.tag.rsplit("}", 1)[-1].lower() == "svg"


def _read_validated_image(path: Path) -> tuple[str, bytes] | None:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if mime not in {*_RASTER_MIMES.values(), _SVG_MIME}:
        return None

    with path.open("rb") as source:
        data = source.read(_IMAGE_MAX_BYTES + 1)
    if not data or len(data) > _IMAGE_MAX_BYTES:
        return None

    valid = _is_decodable_svg(data) if mime == _SVG_MIME else _is_decodable_raster(data, mime)
    if not valid:
        return None
    return mime, data


async def image_attach(params: dict[str, Any]) -> dict[str, Any]:
    parsed = ImageAttachParams.model_validate(params)
    if not parsed.session_id:
        raise ValueError("session_id is required")
    if not parsed.path:
        raise ValueError("path is required")

    attachment_token = _attachment_token(parsed.session_id)
    path = Path(parsed.path.strip().strip("`\"'")).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"attachment is not a file: {path}")
    validated = await asyncio.to_thread(_read_validated_image, path)
    if validated is None:
        raise ValueError(f"attachment is not an image: {path.name}")
    mime, content = validated

    if attachment_token != _attachment_token(parsed.session_id):
        raise ValueError("session attachment state changed during validation; attach again")
    pending = _pending_images.setdefault(parsed.session_id, [])
    if len(pending) >= _IMAGE_MAX_COUNT:
        raise ValueError(f"a turn may have at most {_IMAGE_MAX_COUNT} images")
    if sum(len(item.content or b"") for item in pending) + len(content) > _IMAGE_MAX_TOTAL_BYTES:
        raise ValueError("aggregate image size exceeds the per-turn limit")
    pending.append(Media(path=str(path), mime=mime, kind="image", content=content))
    return {"name": path.name, "remainder": ""}


def pending_images(session_id: str) -> tuple[Media, ...]:
    return tuple(_pending_images.get(session_id, ()))


def consume_pending_images(session_id: str) -> None:
    clear_pending_images(session_id)


def clear_pending_images(session_id: str | None = None) -> None:
    global _attachment_global_generation

    if session_id is None:
        _attachment_global_generation += 1
        _attachment_generations.clear()
        _pending_images.clear()
    else:
        _attachment_generations[session_id] = _attachment_generations.get(session_id, 0) + 1
        _pending_images.pop(session_id, None)


def register_image_methods(dispatcher: "Dispatcher") -> None:
    dispatcher.register("image.attach", image_attach)


__all__ = [
    "clear_pending_images",
    "consume_pending_images",
    "image_attach",
    "pending_images",
    "register_image_methods",
]
