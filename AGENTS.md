# Repository instructions

## Rust extension synchronization

After changing any file under `src-rust/`, `Cargo.toml`, or `Cargo.lock`, do not report the task as complete until the installed Python extension has been rebuilt with:

```bash
uv sync --extra dev --reinstall-package alphanus
```

Then verify that the rebuilt extension loads through the normal application environment:

```bash
uv run pytest -q tests/test_ratatui_pty.py
```

Run relevant Rust formatting and tests before the rebuild. The rebuild must be the final compilation step so `src/alphanus/_alphanus_tui.abi3.so` reflects the latest Rust sources. If Alphanus was already running when the extension was rebuilt, tell the user that the running process must be restarted to load it.
