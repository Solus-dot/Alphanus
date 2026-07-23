import copy
from pathlib import Path
from typing import Any

from alphanus.console import (
    INIT_SECTIONS,
    _CliTheme,
    _print_init_step,
    _print_review_group,
    _print_screen_capture_setup,
    _prompt_choice,
    _prompt_env_name,
    _prompt_with_default,
    _prompt_yes_no,
)
from alphanus.paths import get_app_paths
from core.backend_profiles import BACKEND_PROFILE_LABELS, VALID_BACKEND_PROFILES
from core.configuration import (
    DEFAULT_CONFIG,
    deep_merge,
    load_global_config,
    normalize_config,
    save_global_config,
    validate_endpoint_policy,
)
from core.endpoint_modes import ENDPOINT_MODE_AUTO, ENDPOINT_MODE_CHAT, ENDPOINT_MODE_RESPONSES, ENDPOINT_MODES
from core.search_providers import (
    DEFAULT_TAVILY_API_KEY_ENV,
    SEARCH_FALLBACK_PROVIDERS,
    SEARCH_PROVIDER_SEARXNG,
    SEARCH_PROVIDER_TAVILY,
    SEARCH_PROVIDERS,
)
from core.theme_catalog import DEFAULT_THEME_ID, normalize_theme_id
from core.themes import available_theme_ids, theme_payload


def _run_init(args: Any) -> int:
    theme = _CliTheme()
    section = str(getattr(args, "section", "all") or "all").strip().lower()
    if section not in INIT_SECTIONS:
        section = "all"
    reset_requested = bool(getattr(args, "reset", False))
    app_paths = get_app_paths()
    if str(getattr(args, "api_key", "") or "").strip() or str(getattr(args, "tavily_api_key", "") or "").strip():
        print(
            theme.error(
                "init failed: secret values cannot be passed on the command line; export the configured environment variable instead"
            )
        )
        return 2
    state_root = Path(app_paths.state_root)
    state_root.mkdir(parents=True, exist_ok=True)

    base: dict[str, Any] = copy.deepcopy(dict(DEFAULT_CONFIG))
    existing_warnings: list[str] = []
    if app_paths.config_path.exists():
        try:
            existing = load_global_config(app_paths.config_path, warnings=existing_warnings)
            base = deep_merge(base, existing)
        except (OSError, ValueError):
            base = copy.deepcopy(dict(DEFAULT_CONFIG))
    if reset_requested:
        if section == "all":
            base = copy.deepcopy(dict(DEFAULT_CONFIG))
        else:
            default_cfg = copy.deepcopy(dict(DEFAULT_CONFIG))
            if section in {"all", "model"}:
                current_agent = base.get("agent", {}) if isinstance(base.get("agent"), dict) else {}
                default_agent = default_cfg.get("agent", {}) if isinstance(default_cfg.get("agent"), dict) else {}
                for key in (
                    "base_url",
                    "responses_endpoint",
                    "model_endpoint",
                    "models_endpoint",
                    "endpoint_mode",
                    "backend_profile",
                    "api_key",
                    "api_key_env",
                    "auth_header_template",
                ):
                    if key in default_agent:
                        current_agent[key] = copy.deepcopy(default_agent[key])
                base["agent"] = current_agent
            if section in {"all", "search"}:
                base["search"] = copy.deepcopy(default_cfg.get("search", {}))
            if section in {"all", "theme"}:
                current_tui = base.get("tui", {}) if isinstance(base.get("tui"), dict) else {}
                default_tui = default_cfg.get("tui", {}) if isinstance(default_cfg.get("tui"), dict) else {}
                if "theme" in default_tui:
                    current_tui["theme"] = copy.deepcopy(default_tui["theme"])
                base["tui"] = current_tui

    base_url_default = str(base.get("agent", {}).get("base_url", DEFAULT_CONFIG["agent"]["base_url"]))
    model_endpoint_default = str(base.get("agent", {}).get("model_endpoint", DEFAULT_CONFIG["agent"]["model_endpoint"]))
    responses_endpoint_default = str(base.get("agent", {}).get("responses_endpoint", DEFAULT_CONFIG["agent"]["responses_endpoint"]))
    models_endpoint_default = str(base.get("agent", {}).get("models_endpoint", DEFAULT_CONFIG["agent"]["models_endpoint"]))
    endpoint_mode_default = str(base.get("agent", {}).get("endpoint_mode", DEFAULT_CONFIG["agent"]["endpoint_mode"]))
    backend_profile_default = str(base.get("agent", {}).get("backend_profile", DEFAULT_CONFIG["agent"].get("backend_profile", "auto")))
    api_key_ref_default = str(base.get("agent", {}).get("api_key", DEFAULT_CONFIG["agent"]["api_key"]))
    api_key_env_default = str(base.get("agent", {}).get("api_key_env", DEFAULT_CONFIG["agent"]["api_key_env"]))
    search_provider_default = str(base.get("search", {}).get("provider", DEFAULT_CONFIG["search"]["provider"]))
    search_fallback_default = str(base.get("search", {}).get("fallback_provider", DEFAULT_CONFIG["search"]["fallback_provider"])) or "none"
    tavily_api_key_env_default = str(
        base.get("search", {}).get(
            "tavily_api_key_env",
            DEFAULT_CONFIG["search"].get("tavily_api_key_env", DEFAULT_TAVILY_API_KEY_ENV),
        )
    )
    theme_default = str(base.get("tui", {}).get("theme", DEFAULT_THEME_ID))
    theme_ids = available_theme_ids()
    theme_default, _ = normalize_theme_id(theme_default, default=DEFAULT_THEME_ID, available=theme_ids)

    base_url = base_url_default
    model_endpoint = model_endpoint_default
    responses_endpoint = responses_endpoint_default
    models_endpoint = models_endpoint_default
    endpoint_mode = endpoint_mode_default
    backend_profile = backend_profile_default
    api_key_env = api_key_env_default
    api_key_ref = api_key_ref_default
    search_provider = search_provider_default
    search_fallback_provider = search_fallback_default
    searxng_base_url_default = str(base.get("search", {}).get("searxng_base_url", DEFAULT_CONFIG["search"]["searxng_base_url"]))
    searxng_base_url = searxng_base_url_default
    tavily_api_key_env = tavily_api_key_env_default
    ui_theme = theme_default

    if args.non_interactive:
        print(theme.brand(" ALPHANUS INIT "))
        print(theme.muted(f"Applying non-interactive setup profile ({section})."))
        if section in {"all", "model"}:
            base_url = str(getattr(args, "base_url", "") or "").strip() or base_url_default
            model_endpoint = str(getattr(args, "model_endpoint", "") or "").strip() or model_endpoint_default
            responses_endpoint = str(getattr(args, "responses_endpoint", "") or "").strip() or responses_endpoint_default
            models_endpoint = str(getattr(args, "models_endpoint", "") or "").strip() or models_endpoint_default
            endpoint_mode = str(getattr(args, "endpoint_mode", "") or "").strip() or endpoint_mode_default
            backend_profile = str(getattr(args, "backend_profile", "") or "").strip() or backend_profile_default
            api_key_env = str(getattr(args, "api_key_env", "") or "").strip() or api_key_env_default
            api_key_ref = f"env:{api_key_env}"
        if section in {"all", "search"}:
            search_provider = str(getattr(args, "search_provider", "") or "").strip() or search_provider_default
            search_fallback_provider = str(getattr(args, "search_fallback_provider", "") or "").strip() or search_fallback_default
            searxng_base_url = str(getattr(args, "searxng_base_url", "") or "").strip() or searxng_base_url_default
            tavily_api_key_env = str(getattr(args, "tavily_api_key_env", "") or "").strip() or tavily_api_key_env_default
            if search_provider == SEARCH_PROVIDER_TAVILY:
                search_fallback_provider = "none"
                searxng_base_url = ""
        if section in {"all", "theme"}:
            requested_theme = str(getattr(args, "theme", "") or "").strip() or theme_default
            ui_theme, _ = normalize_theme_id(requested_theme, default=theme_default, available=theme_ids)
    else:
        steps = [name for name in ("model", "search", "theme", "permissions") if section in {"all", name}]
        total_steps = max(len(steps), 1)
        step_index = 1
        print(theme.brand(" ALPHANUS SETUP "))
        print(theme.rule())
        print(theme.muted(f"Wizard scope: {section}. Press Enter to keep defaults."))
        print(theme.muted("We'll configure runtime state, model connectivity, search, and display theme defaults."))
        print(f"{theme.label('State root:')} {theme.path(str(state_root))}")
        print("")
        if section in {"all", "model"}:
            _print_init_step(theme, step_index, total_steps, "Model endpoint")
            use_local = _prompt_yes_no(
                "Use the local OpenAI-compatible endpoint preset (127.0.0.1:8080)?",
                default=base_url_default == str(DEFAULT_CONFIG["agent"]["base_url"]),
                theme=theme,
            )
            if use_local:
                base_url = str(DEFAULT_CONFIG["agent"]["base_url"])
                model_endpoint = str(DEFAULT_CONFIG["agent"]["model_endpoint"])
                responses_endpoint = str(DEFAULT_CONFIG["agent"]["responses_endpoint"])
                models_endpoint = str(DEFAULT_CONFIG["agent"]["models_endpoint"])
                endpoint_mode = str(DEFAULT_CONFIG["agent"]["endpoint_mode"])
                backend_profile = str(DEFAULT_CONFIG["agent"].get("backend_profile", "auto"))
                print(theme.muted("Applied local preset for /v1/responses, /v1/chat/completions, and /v1/models."))
            else:
                base_url = _prompt_with_default(
                    "Base URL",
                    base_url_default,
                    hint=theme.muted("provider root, e.g. https://api.openai.com"),
                    theme=theme,
                )
                model_endpoint = _prompt_with_default(
                    "Chat endpoint",
                    model_endpoint_default,
                    hint=theme.muted("chat completions endpoint"),
                    theme=theme,
                )
                responses_endpoint = _prompt_with_default(
                    "Responses endpoint",
                    responses_endpoint_default,
                    hint=theme.muted("responses endpoint"),
                    theme=theme,
                )
                models_endpoint = _prompt_with_default(
                    "Models endpoint",
                    models_endpoint_default,
                    hint=theme.muted("model catalog endpoint"),
                    theme=theme,
                )
                endpoint_mode = _prompt_choice(
                    theme,
                    "Endpoint mode:",
                    [
                        (ENDPOINT_MODE_AUTO, "responses first with fallback to chat"),
                        (ENDPOINT_MODE_RESPONSES, "force /v1/responses"),
                        (ENDPOINT_MODE_CHAT, "force /v1/chat/completions"),
                    ],
                    default=endpoint_mode_default if endpoint_mode_default in ENDPOINT_MODES else ENDPOINT_MODE_AUTO,
                )
                backend_profile = _prompt_choice(
                    theme,
                    "Backend profile:",
                    [(profile, BACKEND_PROFILE_LABELS.get(profile, profile)) for profile in sorted(VALID_BACKEND_PROFILES)],
                    default=(backend_profile_default if backend_profile_default in VALID_BACKEND_PROFILES else "auto"),
                )
            api_key_env = _prompt_env_name(
                theme,
                "Alphanus API key env var name",
                api_key_env_default,
                hint=theme.muted("name only, not the key value; where Alphanus reads the key"),
            )
            api_key_ref = f"env:{api_key_env.strip() or 'ALPHANUS_API_KEY'}"
            print(theme.muted(f"Export {api_key_env} in your shell before starting Alphanus."))
            step_index += 1
            print("")
        if section in {"all", "search"}:
            _print_init_step(theme, step_index, total_steps, "Search")
            search_provider = _prompt_choice(
                theme,
                "Primary provider:",
                [
                    ("searxng", "local/private search when a SearXNG instance is running"),
                    ("tavily", "hosted fallback search using TAVILY_API_KEY"),
                ],
                default=search_provider_default if search_provider_default in SEARCH_PROVIDERS else SEARCH_PROVIDER_SEARXNG,
            )
            if search_provider == SEARCH_PROVIDER_SEARXNG:
                searxng_base_url = _prompt_with_default(
                    "SearXNG base URL",
                    searxng_base_url_default,
                    hint=theme.muted("used by SearXNG, e.g. http://127.0.0.1:8888"),
                    theme=theme,
                )
                search_fallback_provider = _prompt_choice(
                    theme,
                    "Fallback provider:",
                    [
                        ("tavily", "use TAVILY_API_KEY if SearXNG is unavailable"),
                        ("none", "do not use a hosted fallback"),
                    ],
                    default=search_fallback_default if search_fallback_default in SEARCH_FALLBACK_PROVIDERS else SEARCH_PROVIDER_TAVILY,
                )
            else:
                searxng_base_url = ""
                search_fallback_provider = "none"
            if search_provider == SEARCH_PROVIDER_TAVILY or search_fallback_provider == SEARCH_PROVIDER_TAVILY:
                tavily_api_key_env = _prompt_env_name(
                    theme,
                    "Tavily API key env var name",
                    tavily_api_key_env_default or DEFAULT_TAVILY_API_KEY_ENV,
                    hint=theme.muted("name only, not the key value; where Alphanus reads the Tavily key"),
                )
                print(theme.muted(f"Export {tavily_api_key_env} in your shell before starting Alphanus."))
            step_index += 1
            print("")
        if section in {"all", "theme"}:
            _print_init_step(theme, step_index, total_steps, "Theme")
            theme_options = [(name, str(theme_payload(name).get("description") or "")) for name in theme_ids]
            selected_theme = _prompt_choice(
                theme,
                "Choose a UI theme:",
                theme_options,
                default=theme_default,
            )
            ui_theme, _ = normalize_theme_id(selected_theme, default=theme_default, available=theme_ids)
            step_index += 1
            print("")
        if section in {"all", "permissions"}:
            _print_init_step(theme, step_index, total_steps, "Permissions")
            _print_screen_capture_setup(theme, interactive=True)
            print("")

    updates: dict[str, Any] = {}
    if section in {"all", "model"}:
        updates["agent"] = {
            "base_url": base_url,
            "model_endpoint": model_endpoint,
            "responses_endpoint": responses_endpoint,
            "models_endpoint": models_endpoint,
            "endpoint_mode": endpoint_mode,
            "backend_profile": backend_profile,
            "api_key": api_key_ref,
            "api_key_env": api_key_env,
        }
    if section in {"all", "search"}:
        updates["search"] = {
            "provider": search_provider,
            "fallback_provider": search_fallback_provider,
            "searxng_base_url": searxng_base_url,
            "tavily_api_key_env": tavily_api_key_env,
        }
    if section in {"all", "theme"}:
        updates["tui"] = {"theme": ui_theme}
    merged = deep_merge(base, updates)
    try:
        normalized, warnings = normalize_config(merged)
        validate_endpoint_policy(normalized)
    except ValueError as exc:
        print(f"{theme.error('init failed:')} {exc}")
        return 2

    if not args.non_interactive:
        print(theme.rule("Review"))
        _print_review_group(
            theme,
            "Model",
            [
                ("Base URL", str(normalized["agent"]["base_url"])),
                ("Responses", str(normalized["agent"]["responses_endpoint"])),
                ("Chat", str(normalized["agent"]["model_endpoint"])),
                ("Models", str(normalized["agent"]["models_endpoint"])),
                ("Mode", str(normalized["agent"]["endpoint_mode"])),
                ("Backend", str(normalized["agent"]["backend_profile"])),
                ("API key", str(normalized["agent"]["api_key"])),
            ],
        )
        _print_review_group(
            theme,
            "Search",
            [
                ("Provider", str(normalized["search"]["provider"])),
                ("Fallback", str(normalized["search"]["fallback_provider"] or "none")),
                ("SearXNG", str(normalized["search"]["searxng_base_url"] or "(not set)")),
                ("Tavily env", str(normalized["search"]["tavily_api_key_env"])),
            ],
        )
        _print_review_group(
            theme,
            "Interface",
            [
                ("Theme", str(normalized["tui"]["theme"])),
                ("Secrets", "environment variables only"),
            ],
        )
        if not _prompt_yes_no("Write these settings now?", default=True, theme=theme):
            print(theme.warn("Setup cancelled. No files were written."))
            return 1

    save_global_config(app_paths.config_path, normalized)

    for warning in existing_warnings + warnings:
        print(f"{theme.warn('config warning:')} {warning}")
    if args.non_interactive and section in {"all", "permissions"}:
        _print_screen_capture_setup(theme, interactive=False)
    print("")
    print(theme.ok("Initialization complete."))
    print(f"  {theme.label('Config:')} {theme.path(str(app_paths.config_path))}")
    print(f"  {theme.label('Secrets:')} environment variables only")
    print(f"  {theme.label('Sessions:')} {theme.path(str((Path(app_paths.state_root) / 'sessions').resolve()))}")
    print(f"  {theme.label('Memory:')} {theme.path(str((Path(app_paths.state_root) / 'memory').resolve()))}")
    print("")
    print(theme.accent("Next steps"))
    print(f"  {theme.step('uv run alphanus doctor')}  {theme.muted('validate configuration and dependencies')}")
    print(f"  {theme.step('uv run alphanus')}         {theme.muted('launch the interface')}")
    return 0
