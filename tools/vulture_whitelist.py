# ruff: noqa: F821

# Names in this file are intentionally referenced by external frameworks,
# dynamic loaders, or schema consumers. Keep this whitelist narrow; entries
# here should describe public contracts that static analysis cannot see.

# Bundled skill modules export TOOL_SPECS for the skill registry.
TOOL_SPECS

# Provider/search dataclass fields are serialized through asdict().
freshness_intent
source_preference
latency_ms
result_count

# Config/schema fields are consumed by runtime normalization and user config.
recall_min_score_default
replace_min_score_default
auto_capture
cache_first
min_usable_results
fetch_min_chars
store_path
web_ttl_hours
embeddings_enabled

# TypedDict optional message fields are populated by provider payloads.
video_url

# sqlite3.Connection row_factory is assigned for row access by name.
_.row_factory

# SkillContext fields are populated by turn construction and consumed across
# process/module boundaries.
memory_hits
explicit_skill_args

# urllib calls HTTPRedirectHandler.redirect_request by name.
_.redirect_request

# Textual discovers these attributes and lifecycle/action/event methods by name.
TITLE
CSS
_.compose
_.on_mount
_.get_theme_variable_defaults
_.highlight
_.watch_streaming
_.watch_thinking
_.on_input_changed
_.on_attach_file_pressed
_.on_command_option_selected
_.on_mouse_scroll_up
_.on_mouse_scroll_down
_.action_clear_all
_.action_kill_to_end
_.action_focus_input
_.action_open_command_palette
_.action_open_file_picker
_.action_toggle_details
_.action_toggle_thinking
_.action_toggle_sidebar
_.action_focus_next_panel
_.action_focus_prev_panel
_.action_focus_chat
_.action_focus_tree
_.action_tree_prev_sibling
_.action_tree_next_sibling
_.action_scroll_up
_.action_scroll_down
_.action_move_down
_.action_move_up
_._copy_all
_._close
_._save_button
_._cancel_button
_._close_button
_._open_button
_._delete_button
_._new_button
_._create_button
_._confirm_button
_._option_selected
_._session_query_changed
_._session_query_submitted
_._name_submitted
_._query_changed
_._query_submitted
_._open_config_editor
_._cmd_help
_._cmd_tree
_.placeholder
_.cursor_location
_.variant
_._debug_mode
_._shell_confirm_command

# Test doubles and live preview state are accessed via framework/runtime hooks.
_.mark_rendered_filepaths
