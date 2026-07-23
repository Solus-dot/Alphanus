from pathlib import Path
from typing import Any

from alphanus.console import _CliTheme, _prompt_yes_no
from alphanus.paths import get_app_paths
from alphanus.runtime_factory import _load_runtime_config


def _run_retrieval(args: Any) -> int:
    from core.retrieval import SQLiteRetrievalStore, configured_store_path

    app_paths = get_app_paths()
    theme = _CliTheme()
    try:
        config, warnings = _load_runtime_config(app_paths, args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"retrieval failed: {exc}")
        return 2
    for warning in warnings:
        print(f"{theme.warn('config warning:')} {warning}")

    db_path = configured_store_path(config)
    command = str(getattr(args, "retrieval_command", "") or "stats")
    if command == "reset":
        if not bool(getattr(args, "yes", False)) and not _prompt_yes_no(f"Delete retrieval store at {db_path}?", default=False):
            print(theme.warn("Reset cancelled."))
            return 1
        deleted = []
        for suffix in ("", "-wal", "-shm"):
            target = Path(f"{db_path}{suffix}")
            if target.exists():
                target.unlink()
                deleted.append(str(target))
        SQLiteRetrievalStore(db_path)
        print(theme.ok("Retrieval store reset."))
        if deleted:
            for path in deleted:
                print(f"  {theme.path(path)}")
        return 0

    stats = SQLiteRetrievalStore(db_path).stats()
    print(theme.brand(" ALPHANUS RETRIEVAL "))
    print(f"  {theme.label('Store:')} {theme.path(str(db_path))}")
    print(f"  {theme.label('Records:')} {stats['records']}")
    print(f"  {theme.label('Chunks:')} {stats['chunks']}")
    print(f"  {theme.label('Stale web records:')} {stats['stale_records']}")
    by_type = stats.get("by_type", {})
    if isinstance(by_type, dict) and by_type:
        print(f"  {theme.label('By type:')} " + ", ".join(f"{key}={value}" for key, value in sorted(by_type.items())))
    return 0
