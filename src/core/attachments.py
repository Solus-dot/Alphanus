from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

DEFAULT_IMAGE_MIME = "image/jpeg"
MAX_ATTACHMENTS = 10
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_TEXT_ATTACHMENT_BYTES = 1024 * 1024
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _validated_path(path: str, *, max_bytes: int = MAX_ATTACHMENT_BYTES) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"Attachment symlinks are not allowed: {candidate.name}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Attachment is not a regular file: {candidate.name}")
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Attachment exceeds {max_bytes} bytes: {candidate.name}")
    return resolved


def image_mime_type(path: str) -> str | None:
    mime, _encoding = mimetypes.guess_type(path)
    return mime if mime and mime.startswith("image/") else None


def classify_attachment(path: str) -> str:
    resolved = _validated_path(path)
    mime = image_mime_type(path)
    if mime:
        if mime not in ALLOWED_IMAGE_MIMES:
            raise ValueError(f"Unsupported image type: {mime}")
        return "image"
    try:
        with resolved.open("rb") as handle:
            data = handle.read(MAX_TEXT_ATTACHMENT_BYTES + 1)
        if len(data) > MAX_TEXT_ATTACHMENT_BYTES:
            return "unknown"
        data.decode("utf-8")
    except Exception:
        return "unknown"
    return "unknown" if b"\x00" in data else "text"


def encode_image(path: str) -> tuple[str, str]:
    mime = image_mime_type(path) or DEFAULT_IMAGE_MIME
    if mime not in ALLOWED_IMAGE_MIMES:
        raise ValueError(f"Unsupported image type: {mime}")
    resolved = _validated_path(path)
    with resolved.open("rb") as handle:
        data = base64.b64encode(handle.read()).decode("utf-8")
    return data, mime


def read_text_file(path: str) -> str:
    resolved = _validated_path(path, max_bytes=MAX_TEXT_ATTACHMENT_BYTES)
    return resolved.read_text(encoding="utf-8")


def build_content(text: str, attachments: list[tuple[str, str]]):
    if not attachments:
        return text

    if len(attachments) > MAX_ATTACHMENTS:
        raise ValueError(f"At most {MAX_ATTACHMENTS} attachments are allowed")
    total_bytes = sum(_validated_path(path).stat().st_size for path, _kind in attachments)
    if total_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
        raise ValueError(f"Attachments exceed the {MAX_TOTAL_ATTACHMENT_BYTES}-byte aggregate limit")

    image_parts = []
    prefix = ""
    summary_items: list[str] = []
    for path, kind in attachments:
        name = os.path.basename(path)
        summary_items.append(f"{name} ({kind})")
        if kind == "image":
            data, mime = encode_image(path)
            image_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        elif kind == "text":
            body = read_text_file(path)
            ext = Path(path).suffix.lstrip(".")
            fence = f"```{ext}\n{body}\n```" if ext else f"```\n{body}\n```"
            prefix += f"[File: {name}]\n{fence}\n\n"

    summary = f"[Attachments: {', '.join(summary_items)}]\n\n" if summary_items else ""
    parts = [{"type": "text", "text": summary + prefix + text}]
    parts.extend(image_parts)
    return parts
