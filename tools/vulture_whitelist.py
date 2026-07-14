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
