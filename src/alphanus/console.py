import os
import platform
import re
import shutil
import subprocess
import sys
from functools import partialmethod
from pathlib import Path

INIT_SECTIONS = ("all", "model", "search", "theme", "permissions")
_VALID_CLI_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _CliTheme:
    def __init__(self) -> None:
        term = os.environ.get("TERM", "").lower()
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)()) and os.environ.get("NO_COLOR") is None and term != "dumb"

    def _fmt(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def _fg(self, text: str, rgb: tuple[int, int, int], *, bold: bool = False, dim: bool = False) -> str:
        parts = []
        if bold:
            parts.append("1")
        if dim:
            parts.append("2")
        parts.append(f"38;2;{rgb[0]};{rgb[1]};{rgb[2]}")
        return self._fmt(text, ";".join(parts))

    accent = partialmethod(_fg, rgb=(99, 102, 241), bold=True)
    cyan_text = partialmethod(_fg, rgb=(34, 211, 238), bold=True)
    value = partialmethod(_fg, rgb=(167, 139, 250), bold=True)
    muted = partialmethod(_fg, rgb=(161, 161, 170), dim=True)
    ok = partialmethod(_fg, rgb=(16, 185, 129), bold=True)
    warn = partialmethod(_fg, rgb=(245, 158, 11), bold=True)
    error = partialmethod(_fg, rgb=(244, 63, 94), bold=True)
    label = partialmethod(_fg, rgb=(228, 228, 231), bold=True)
    path = partialmethod(_fg, rgb=(199, 210, 254))

    def brand(self, text: str) -> str:
        return self._fmt(text, "1;38;2;199;210;254;48;2;49;46;129")

    def step(self, text: str) -> str:
        return f"{self.accent('>')} {self.label(text)}"

    def rule(self, title: str = "") -> str:
        if title:
            return f"{self.muted('--')} {self.accent(title)} {self.muted('-' * max(10, 48 - len(title)))}"
        return self.muted("-" * 52)


def _prompt_with_default(label: str, default: str, *, hint: str = "", theme: _CliTheme | None = None) -> str:
    default_text = theme.value(default) if theme is not None else default
    prompt = f"{label} [{default_text}]"
    prompt = f"{prompt}\n  {hint}" if hint else prompt
    return input(f"{prompt}: ").strip() or default


def _prompt_env_name(theme: _CliTheme, label: str, default: str, *, hint: str = "") -> str:
    while True:
        value = _prompt_with_default(label, default, hint=hint, theme=theme).strip()
        if not value:
            return ""
        lowered = value.lower()
        looks_like_secret_value = lowered.startswith(("sk-", "tvly-", "bearer ", "key-")) or (
            len(value) >= 24 and any(char.isdigit() for char in value) and any(char.isalpha() for char in value)
        )
        if looks_like_secret_value:
            print(theme.warn("Enter an environment variable name here, not the API key value."))
            continue
        if not _VALID_CLI_ENV_NAME_RE.match(value):
            print(theme.warn("Environment variable names must use letters, numbers, and underscores, and cannot start with a number."))
            continue
        if value != value.upper():
            print(theme.warn("Use an uppercase environment variable name here, such as ALPHANUS_API_KEY or OPENAI_API_KEY."))
            continue
        return value


def _prompt_yes_no(label: str, *, default: bool = True, theme: _CliTheme | None = None) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    suffix = theme.value(suffix) if theme is not None else suffix
    raw = input(f"{label} {suffix}: ").strip().lower()
    return {"y": True, "yes": True, "n": False, "no": False}.get(raw, default)


def _prompt_choice(theme: _CliTheme, label: str, options: list[tuple[str, str]], *, default: str) -> str:
    print(theme.label(label))
    value_width = max((len(value) for value, _desc in options), default=7)
    for idx, (value, description) in enumerate(options, start=1):
        marker = theme.ok("*") if value == default else " "
        print(f"  {marker} {theme.accent(str(idx).rjust(2) + '.')} {theme.value(f'{value:<{value_width}}')}  {theme.muted(description)}")
    default_index = next((idx for idx, (value, _desc) in enumerate(options, start=1) if value == default), 1)
    while True:
        raw = input(f"{theme.cyan_text('Choose option')} {theme.value('[' + str(default_index) + ']')}: ").strip().lower()
        if not raw:
            return default
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        for value, _desc in options:
            if raw == value:
                return value
        print(theme.warn("Invalid choice. Enter option number or name."))


def _screen_capture_setup_lines(*, open_settings: bool = False) -> list[tuple[str, str]]:
    system = platform.system().lower()
    if system == "darwin":
        if open_settings:
            subprocess.run(
                ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return [
            ("macOS Screen Recording", "required for screenshot capture"),
            ("Setup", "System Settings > Privacy & Security > Screen Recording"),
            ("Allow", "the terminal app or launcher used to run Alphanus, then restart it"),
        ]
    if system == "linux":
        binary = shutil.which("gnome-screenshot") or shutil.which("scrot")
        session = os.environ.get("XDG_SESSION_TYPE", "").strip().lower() or "unknown"
        detail = f"found {Path(binary).name}; session={session}" if binary else "install gnome-screenshot or scrot"
        if session == "wayland":
            detail += "; compositor/portal permission prompts may still be required"
        return [("Linux screenshot helper", detail)]
    return [(f"{platform.system() or 'Unknown'} screenshot helper", "unsupported by screenshot-ocr")]


def _print_screen_capture_setup(theme: _CliTheme, *, interactive: bool) -> None:
    open_settings = False
    if interactive and platform.system().lower() == "darwin":
        open_settings = _prompt_yes_no("Open macOS Screen Recording settings now?", default=True, theme=theme)
    print("")
    print(theme.accent("Screen Capture Permissions"))
    for label, detail in _screen_capture_setup_lines(open_settings=open_settings):
        print(f"  {theme.label(label + ':')} {detail}")


def _print_init_step(theme: _CliTheme, index: int, total: int, title: str) -> None:
    print(theme.rule(f"Step {index}/{total}"))
    print(theme.cyan_text(title))
    print("")


def _print_review_group(theme: _CliTheme, title: str, rows: list[tuple[str, str]]) -> None:
    print(f"  {theme.accent(title)}")
    label_width = max((len(label) for label, _value in rows), default=8)
    for label, value in rows:
        print(f"    {theme.label((label + ':').ljust(label_width + 1))} {theme.value(value)}")
    print("")


def _doctor_state(theme: _CliTheme, state: str) -> str:
    return {"ok": theme.ok("[OK]"), "wait": theme.warn("[WAIT]")}.get(state.strip().lower(), theme.error("[FAIL]"))


def _print_doctor_group(theme: _CliTheme, title: str, rows: list[tuple[str, str, str]]) -> None:
    print(f"  {theme.accent(title)}")
    label_width = max((len(label) for label, _state, _detail in rows), default=8)
    for label, state, detail in rows:
        line = f"    {theme.label((label + ':').ljust(label_width + 1))} {_doctor_state(theme, state)}"
        if detail:
            line = f"{line}  {theme.muted(detail)}"
        print(line)
    print("")
