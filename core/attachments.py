from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_MIME: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

TEXT_EXTS = {
    ".txt",
    ".md",
    ".markdown",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".sql",
    ".html",
    ".css",
    ".sh",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rb",
    ".php",
    ".swift",
    ".xml",
    ".env",
    ".log",
}


def classify_attachment(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_MIME:
        return "image"
    if ext in TEXT_EXTS or ext == "":
        try:
            with open(path, "rb") as handle:
                handle.read(512).decode("utf-8")
            return "text"
        except Exception:
            return "unknown"
    return "unknown"


def encode_image(path: str) -> Tuple[str, str]:
    ext = Path(path).suffix.lower()
    mime = IMAGE_MIME.get(ext, "image/jpeg")
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("utf-8")
    return data, mime


def read_text_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="latin-1")


def build_content(text: str, attachments: List[Tuple[str, str]]):
    if not attachments:
        return text

    parts = []
    prefix = ""
    for path, kind in attachments:
        name = os.path.basename(path)
        if kind == "image":
            data, mime = encode_image(path)
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        elif kind == "text":
            body = read_text_file(path)
            ext = Path(path).suffix.lstrip(".")
            fence = f"```{ext}\n{body}\n```" if ext else f"```\n{body}\n```"
            prefix += f"[File: {name}]\n{fence}\n\n"

    parts.append({"type": "text", "text": prefix + text})
    return parts
