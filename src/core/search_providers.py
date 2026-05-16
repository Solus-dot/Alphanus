from __future__ import annotations

from typing import Final

SEARCH_PROVIDER_SEARXNG: Final[str] = "searxng"
SEARCH_PROVIDER_TAVILY: Final[str] = "tavily"
SEARCH_FALLBACK_NONE: Final[str] = "none"
DEFAULT_TAVILY_API_KEY_ENV: Final[str] = "TAVILY_API_KEY"

SEARCH_PROVIDERS: Final[tuple[str, ...]] = (SEARCH_PROVIDER_SEARXNG, SEARCH_PROVIDER_TAVILY)
SEARCH_FALLBACK_PROVIDERS: Final[tuple[str, ...]] = (SEARCH_FALLBACK_NONE, SEARCH_PROVIDER_TAVILY)
