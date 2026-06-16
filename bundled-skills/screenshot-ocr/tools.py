from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from skills.runtime import ToolExecutionEnv

_DEFAULT_OCR_MAX_CHARS = 20_000
_MAX_OCR_CHARS_LIMIT = 1_000_000
_SCREENSHOT_TIMEOUT_S = 20
_TESSERACT_TIMEOUT_S = 60
_DEFAULT_SCREENSHOT_FILENAME = "screenshot.png"

TOOL_SPECS = {
    "capture_screenshot": {
        "capability": "screen_capture",
        "mutates": True,
        "actions": ["read", "check"],
        "description": "Capture a full-screen screenshot. Requires confirm_capture=true.",
        "parameters": {
            "type": "object",
            "properties": {"output_path": {"type": "string"}, "confirm_capture": {"type": "boolean"}},
            "required": [],
        },
    },
    "ocr_image": {
        "capability": "ocr_read",
        "mutates": False,
        "actions": ["read", "check"],
        "description": "Run OCR on an explicit image file path. Requires optional OCR tooling.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}},
            "required": ["path"],
        },
    },
    "capture_and_ocr": {
        "capability": "screen_capture",
        "mutates": True,
        "actions": ["read", "check"],
        "description": "Capture a full-screen screenshot and OCR it. Requires confirm_capture=true and OCR tooling.",
        "parameters": {
            "type": "object",
            "properties": {"max_chars": {"type": "integer"}, "confirm_capture": {"type": "boolean"}},
            "required": [],
        },
    },
}


def _ok(data: dict[str, object]) -> dict[str, object]:
    return {"ok": True, "data": data, "error": None, "meta": {}}


def _err(code: str, message: str, data: dict[str, object] | None = None) -> dict[str, object]:
    return {"ok": False, "data": data or {}, "error": {"code": code, "message": message}, "meta": {}}


def _resolve_read_path(raw: str, env: ToolExecutionEnv) -> Path:
    if not raw.strip():
        raise ValueError("path is required")
    path = env.workspace._resolve_read_path(raw)  # noqa: SLF001
    if env.workspace._is_secret_path(path):  # noqa: SLF001
        raise PermissionError("Image path matches sensitive file policy")
    if env.workspace._is_protected_state_path(path):  # noqa: SLF001
        raise PermissionError("Image path targets protected internal state")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {raw}")
    return path


def _resolve_output(raw: str, env: ToolExecutionEnv) -> Path:
    workspace = Path(env.workspace.workspace_root).expanduser().resolve()
    if raw.strip():
        path = env.workspace._resolve_write_path(raw)  # noqa: SLF001
    else:
        path = env.workspace._resolve_write_path(str(workspace / _DEFAULT_SCREENSHOT_FILENAME))  # noqa: SLF001
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _capture(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    if not bool(args.get("confirm_capture")):
        return _err("E_CONFIRMATION_REQUIRED", "Screenshot capture requires confirm_capture=true", {})
    output = _resolve_output(str(args.get("output_path") or ""), env)
    system = platform.system().lower()
    if system == "darwin":
        cmd = ["screencapture", "-x", str(output)]
    elif system == "linux":
        binary = shutil.which("gnome-screenshot") or shutil.which("scrot")
        if binary is None:
            return _err("E_DEPENDENCY", "Screenshot capture on Linux requires gnome-screenshot or scrot", {"platform": system})
        cmd = [binary, "-f", str(output)] if Path(binary).name == "gnome-screenshot" else [binary, str(output)]
    elif system == "windows":
        process_env = os.environ.copy()
        process_env["ALPHANUS_SCREENSHOT_OUTPUT"] = str(output)
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "Add-Type -AssemblyName System.Drawing; "
            "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height; "
            "$g=[System.Drawing.Graphics]::FromImage($bmp); "
            "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size); "
            "$bmp.Save($env:ALPHANUS_SCREENSHOT_OUTPUT);"
        )
        cmd = ["powershell", "-NoProfile", "-Command", script]
    else:
        return _err("E_UNSUPPORTED", f"Unsupported platform: {system}", {"platform": system})
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_SCREENSHOT_TIMEOUT_S,
        env=process_env if system == "windows" else None,
    )
    if proc.returncode != 0:
        return _err("E_IO", proc.stderr.strip() or "Screenshot capture failed", {"platform": system, "output_path": str(output)})
    return _ok({"platform": system, "output_path": str(output)})


def _ocr_file(path: Path, max_chars: int) -> dict[str, object]:
    limit = max(1, min(_MAX_OCR_CHARS_LIMIT, int(max_chars or _DEFAULT_OCR_MAX_CHARS)))
    try:
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        pytesseract = None
        Image = None
    if pytesseract is not None and Image is not None:
        text = str(pytesseract.image_to_string(Image.open(path)))
        return _ok({"path": str(path), "text": text[:limit], "truncated": len(text) > limit, "engine": "pytesseract"})
    binary = shutil.which("tesseract")
    if binary is None:
        return _err("E_DEPENDENCY", "OCR requires pytesseract+Pillow or the tesseract binary", {"path": str(path)})
    with tempfile.TemporaryDirectory() as tmp:
        out_base = Path(tmp) / "ocr"
        proc = subprocess.run([binary, str(path), str(out_base)], capture_output=True, text=True, timeout=_TESSERACT_TIMEOUT_S)
        if proc.returncode != 0:
            return _err("E_IO", proc.stderr.strip() or "tesseract failed", {"path": str(path)})
        text_path = out_base.with_suffix(".txt")
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
    return _ok({"path": str(path), "text": text[:limit], "truncated": len(text) > limit, "engine": "tesseract"})


def _ocr(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    path = _resolve_read_path(str(args.get("path") or ""), env)
    return _ocr_file(path, int(args.get("max_chars") or _DEFAULT_OCR_MAX_CHARS))


def _capture_and_ocr(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    capture = _capture({"confirm_capture": args.get("confirm_capture"), "output_path": ""}, env)
    if not capture.get("ok"):
        return capture
    data = capture.get("data") if isinstance(capture.get("data"), dict) else {}
    path = Path(str(data.get("output_path")))
    ocr = _ocr_file(path, int(args.get("max_chars") or _DEFAULT_OCR_MAX_CHARS))
    if not ocr.get("ok"):
        return ocr
    ocr_data = ocr.get("data") if isinstance(ocr.get("data"), dict) else {}
    ocr_data["screenshot_path"] = str(path)
    return _ok(ocr_data)


def execute(tool_name: str, args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    if tool_name == "capture_screenshot":
        return _capture(args, env)
    if tool_name == "ocr_image":
        return _ocr(args, env)
    if tool_name == "capture_and_ocr":
        return _capture_and_ocr(args, env)
    raise ValueError(f"Unsupported tool: {tool_name}")
