from __future__ import annotations

import base64
import os
from pathlib import Path

IMAGE_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

def classify_attachment(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_MIME:
        return "image"
    try:
        data = Path(path).read_bytes()
        data.decode("utf-8")
    except Exception:
        return "unknown"
    return "unknown" if b"\x00" in data else "text"


def encode_image(path: str) -> tuple[str, str]:
    ext = Path(path).suffix.lower()
    mime = IMAGE_MIME.get(ext, "image/jpeg")
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("utf-8")
    return data, mime


def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_content(text: str, attachments: list[tuple[str, str]]):
    if not attachments:
        return text

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
