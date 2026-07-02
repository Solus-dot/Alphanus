import ipaddress
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html import unescape
from typing import Any

from core.coercion import coerce_bool
from core.endpoint_modes import OPENAI_EMBEDDINGS_PATH
from core.retrieval import SQLiteRetrievalStore, configured_store_path
from core.search_providers import DEFAULT_TAVILY_API_KEY_ENV, SEARCH_PROVIDER_SEARXNG, SEARCH_PROVIDER_TAVILY
from core.streaming import should_retry
from skills.runtime import ToolExecutionEnv

TOOL_SPECS = {
    "web_search": {
        "capability": "web_search",
        "mutates": False,
        "actions": ["read", "check"],
        "description": "Search the public web and return structured results with titles, URLs, snippets, and source metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    "fetch_url": {
        "capability": "web_fetch",
        "mutates": False,
        "actions": ["read"],
        "description": "Fetch a URL and extract readable text content plus source metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["url"],
        },
    },
    "retrieve_knowledge": {
        "capability": "knowledge_retrieve",
        "mutates": False,
        "actions": ["read", "check"],
        "description": "Search the local SQLite retrieval index for web, memory, project, and tool outcome records.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
    "index_project": {
        "capability": "project_index",
        "mutates": True,
        "actions": ["update"],
        "description": "Index explicitly selected project files into the local retrieval store.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}},
                "max_chars_per_file": {"type": "integer"},
            },
            "required": ["paths"],
        },
    },
    "retrieval_stats": {
        "capability": "retrieval_stats",
        "mutates": False,
        "actions": ["check", "read"],
        "description": "Return local retrieval database statistics and embedding availability.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "forget_retrieval_record": {
        "capability": "retrieval_forget",
        "mutates": True,
        "actions": ["delete", "remove"],
        "description": "Delete a retrieval record by id.",
        "parameters": {
            "type": "object",
            "properties": {"record_id": {"type": "integer"}},
            "required": ["record_id"],
        },
    },
}

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
_BLOCK_TAG_RE = re.compile(r"</?(?:p|div|section|article|li|ul|ol|h[1-6]|br|tr|td|th|blockquote)[^>]*>", re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_RE = re.compile(r"<meta\s+([^>]+)>", re.IGNORECASE)
_ATTR_RE = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*("([^"]*)"|\'([^\']*)\'|([^\s>]+))')
_HEADING_RE = re.compile(r"<h([1-3])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_CAPTURE_RE = re.compile(r"<(?:p|li|blockquote|h[1-6]|td|th)[^>]*>(.*?)</(?:p|li|blockquote|h[1-6]|td|th)>", re.IGNORECASE | re.DOTALL)
_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_ALLOWED_FETCH_CONTENT_TYPES = ("text/html", "text/plain", "application/json", "application/xml", "text/xml")
_REDIRECT_HTTP_STATUS = {301, 302, 303, 307, 308}
_PRIVATE_HOST_SUFFIXES = (".local", ".internal", ".lan", ".home.arpa")
_PRIVATE_HOST_LITERALS = {"localhost", "localhost.localdomain", "0.0.0.0", "::", "::1"}
_TRUSTED_SOURCE_HINTS = (
    ".gov",
    ".mil",
    ".edu",
    ".org",
    "wikipedia.org",
    "github.com",
    "official",
    "docs.",
)
_TRACKING_QUERY_KEYS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)
_FAILURE_CONFIG = "config"
_FAILURE_NETWORK = "network"
_FAILURE_RATE_LIMIT = "rate_limit"
_FAILURE_INVALID_RESPONSE = "invalid_response"
_FAILURE_EMPTY_RESULTS = "empty_results"
_FAILURE_UNSUPPORTED = "unsupported_provider"
_FAILURE_FETCH_BLOCKED = "fetch_blocked"
_FAILURE_FETCH_UNUSABLE = "fetch_unusable"


class SearchError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str, recoverable: bool = True) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.recoverable = recoverable


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    limit: int
    freshness_intent: str = "current"
    source_preference: str = "best"


@dataclass(slots=True)
class SearchAttempt:
    provider: str
    status: str
    failure_class: str = ""
    error: str = ""
    latency_ms: int = 0
    result_count: int = 0


@dataclass(slots=True)
class SearchResponse:
    request: SearchRequest
    results: list[dict[str, Any]]
    attempts: list[SearchAttempt] = field(default_factory=list)
    cache_hits: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    degraded: bool = False
    failure_class: str = ""
    evidence_quality: str = "none"

    def to_payload(self) -> dict[str, Any]:
        provider = self.provider or (self.results[0].get("provider", "") if self.results else "")
        payload = {
            "query": self.request.query,
            "results": self.results,
            "provider": provider,
            "search_engine": provider,
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "degraded": self.degraded,
            "failure_class": self.failure_class,
            "evidence_quality": self.evidence_quality,
            "cache_hits": self.cache_hits,
        }
        return payload


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _request(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None) -> urllib.request.Request:
    merged = {"User-Agent": _USER_AGENT}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, data=data, headers=merged)


def _open_no_redirect(req: urllib.request.Request, *, timeout_s: float):
    return _NO_REDIRECT_OPENER.open(req, timeout=timeout_s)


def _decode_response(resp) -> tuple[str, str]:
    raw = resp.read()
    content_type = resp.headers.get("Content-Type", "text/html")
    charset_getter = getattr(resp.headers, "get_content_charset", None)
    charset = charset_getter() if callable(charset_getter) else None
    charset = charset or "utf-8"
    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    return content_type, text


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(str(text or ""))).strip()


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def _canonicalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return ""
    query_items = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_KEYS
    ]
    cleaned = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/") or "/",
        params="",
        query=urllib.parse.urlencode(query_items, doseq=True),
        fragment="",
    )
    return urllib.parse.urlunparse(cleaned)


def _query_tokens(query: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]{2,}", str(query or "").lower())
        if token not in {"the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "which", "who"}
    ]


def _query_match_score(text: str, query: str) -> float:
    tokens = _query_tokens(query)
    if not tokens:
        return 0.0
    hay = str(text or "").lower()
    hits = sum(1 for token in tokens if token in hay)
    return round(min(1.0, hits / max(1, len(tokens))), 2)


def _parse_dateish(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    cleaned = text.replace("Z", "+00:00") if text.endswith("Z") else text
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC).isoformat()
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _freshness_score(published_at: str) -> float:
    if not published_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_days = max(0.0, (datetime.now(UTC) - dt.astimezone(UTC)).total_seconds() / 86400.0)
    return round(max(0.0, 1.0 - min(age_days, 365.0) / 365.0), 2)


def _source_type(host: str, title: str = "", snippet: str = "") -> str:
    hay = " ".join(part for part in (host, title, snippet) if part).lower()
    if any(host.endswith(suffix) for suffix in (".gov", ".mil", ".edu")) or re.search(r"\bofficial\b", hay):
        return "official"
    if "docs." in host or "/docs" in hay:
        return "documentation"
    if host.endswith(".org") or "wikipedia.org" in host or "github.com" in host:
        return "reference"
    if any(token in host for token in ("reuters", "apnews", "bbc", "cnn", "nytimes", "wsj", "bloomberg", "techcrunch", "theverge")):
        return "news"
    return "community"


def _trust_score(host: str, source_type: str) -> float:
    if not host:
        return 0.0
    if any(host.endswith(suffix) for suffix in (".gov", ".mil", ".edu")):
        return 1.0
    if source_type in {"official", "documentation"}:
        return 0.92
    if any(hint in host for hint in _TRUSTED_SOURCE_HINTS):
        return 0.85
    if source_type == "news":
        return 0.75
    if host.count(".") >= 2:
        return 0.6
    return 0.45


def _normalized_snippet(item: dict) -> str:
    for key in ("content", "description", "snippet"):
        value = _clean_text(item.get(key, ""))
        if value:
            return value
    extra = item.get("extra_snippets")
    if isinstance(extra, list):
        for snippet in extra:
            value = _clean_text(snippet)
            if value:
                return value
    return ""


def _raw_description(item: dict) -> str:
    for key in ("description", "content", "snippet"):
        value = _clean_text(item.get(key, ""))
        if value:
            return value
    return ""


def _raw_published_at(item: dict) -> str:
    for key in ("published_date", "published_at", "date", "last_updated"):
        parsed = _parse_dateish(item.get(key))
        if parsed:
            return parsed
    return ""


def _selection_reason(source_type: str, query_match_score: float, freshness_score: float) -> str:
    reasons: list[str] = []
    if source_type in {"official", "documentation"}:
        reasons.append("primary or official source")
    elif source_type == "news":
        reasons.append("news reporting")
    if query_match_score >= 0.7:
        reasons.append("strong query match")
    elif query_match_score >= 0.4:
        reasons.append("relevant query match")
    if freshness_score >= 0.7:
        reasons.append("recent metadata")
    return ", ".join(reasons[:3]) or "best available result"


def _ranking_bonus(source_type: str) -> float:
    if source_type in {"official", "documentation"}:
        return 0.18
    if source_type == "reference":
        return 0.08
    if source_type == "news":
        return 0.04
    return 0.0


def _provider_name(env: ToolExecutionEnv) -> str:
    search_cfg = env.config.get("search", {}) if isinstance(env.config, dict) else {}
    return str(search_cfg.get("provider", SEARCH_PROVIDER_SEARXNG)).strip().lower() or SEARCH_PROVIDER_SEARXNG


def _fallback_provider_name(env: ToolExecutionEnv) -> str:
    search_cfg = env.config.get("search", {}) if isinstance(env.config, dict) else {}
    return str(search_cfg.get("fallback_provider", "")).strip().lower()


def _search_cfg(env: ToolExecutionEnv) -> dict[str, Any]:
    if not isinstance(getattr(env, "config", None), dict):
        return {}
    cfg = env.config.get("search")
    return cfg if isinstance(cfg, dict) else {}


def _cfg_float(search_cfg: dict[str, Any], key: str, default: float, *, minimum: float = 0.0) -> float:
    raw = search_cfg.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _cfg_int(search_cfg: dict[str, Any], key: str, default: int, *, minimum: int = 0) -> int:
    raw = search_cfg.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _cfg_bool(search_cfg: dict[str, Any], key: str, default: bool) -> bool:
    return coerce_bool(search_cfg.get(key, default), default)


def _request_timeout_s(env: ToolExecutionEnv) -> float:
    return _cfg_float(_search_cfg(env), "request_timeout_s", 20.0, minimum=1.0)


def _request_retries(env: ToolExecutionEnv) -> int:
    return _cfg_int(_search_cfg(env), "request_retries", 1, minimum=0)


def _request_retry_backoff_s(env: ToolExecutionEnv) -> float:
    return _cfg_float(_search_cfg(env), "request_retry_backoff_s", 0.5, minimum=0.0)


def _fetch_max_redirects(env: ToolExecutionEnv) -> int:
    return _cfg_int(_search_cfg(env), "fetch_max_redirects", 5, minimum=0)


def _cache_first(env: ToolExecutionEnv) -> bool:
    return _cfg_bool(_search_cfg(env), "cache_first", True)


def _min_usable_results(env: ToolExecutionEnv) -> int:
    return _cfg_int(_search_cfg(env), "min_usable_results", 1, minimum=1)


def _fetch_min_chars(env: ToolExecutionEnv) -> int:
    return _cfg_int(_search_cfg(env), "fetch_min_chars", 20, minimum=1)


def _provider_chain(env: ToolExecutionEnv) -> list[str]:
    cfg = _search_cfg(env)
    raw_chain = cfg.get("provider_chain")
    chain: list[str] = []
    if isinstance(raw_chain, list):
        chain = [str(item).strip().lower() for item in raw_chain if str(item).strip()]
    elif isinstance(raw_chain, str) and raw_chain.strip():
        chain = [item.strip().lower() for item in raw_chain.split(",") if item.strip()]
    if not chain:
        provider = _provider_name(env)
        fallback = _fallback_provider_name(env)
        chain = [provider]
        if fallback and fallback != provider:
            chain.append(fallback)
    out: list[str] = []
    for provider in chain:
        if provider == "none":
            continue
        if provider not in {SEARCH_PROVIDER_SEARXNG, SEARCH_PROVIDER_TAVILY}:
            out.append(provider)
            continue
        if provider not in out:
            out.append(provider)
    return out or [SEARCH_PROVIDER_SEARXNG]


def _web_ttl_seconds(env: ToolExecutionEnv) -> int:
    cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
    raw_hours = cfg.get("web_ttl_hours", 72) if isinstance(cfg, dict) else 72
    try:
        hours = float(raw_hours)
    except (TypeError, ValueError):
        hours = 72.0
    return max(0, int(hours * 3600))


def _retrieval_enabled(env: ToolExecutionEnv) -> bool:
    cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
    return _cfg_bool(cfg, "enabled", True) if isinstance(cfg, dict) else True


def _retrieval_store(env: ToolExecutionEnv) -> SQLiteRetrievalStore:
    return SQLiteRetrievalStore(configured_store_path(env.config if isinstance(env.config, dict) else {}))


def _embedding_cfg(env: ToolExecutionEnv) -> dict[str, Any]:
    retrieval_cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
    embeddings_cfg = retrieval_cfg.get("embeddings", {}) if isinstance(retrieval_cfg, dict) else {}
    return embeddings_cfg if isinstance(embeddings_cfg, dict) else {}


def _embedding_endpoint(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/embeddings") else f"{trimmed}{OPENAI_EMBEDDINGS_PATH}"


def _embedding_vectors(texts: list[str], env: ToolExecutionEnv) -> tuple[str, list[list[float]]]:
    cfg = _embedding_cfg(env)
    if not bool(cfg.get("enabled", False)):
        return "", []
    base_url = str(cfg.get("base_url") or "").strip()
    model = str(cfg.get("model") or "").strip()
    if not base_url or not model or not texts:
        return "", []
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key_env = str(cfg.get("api_key_env") or "").strip()
    api_key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = _request(_embedding_endpoint(base_url), data=payload, headers=headers)
    response = _request_json(req, provider_name="Embeddings", timeout_s=_request_timeout_s(env), retries=0, retry_backoff_s=0.0)
    rows = response.get("data")
    if not isinstance(rows, list):
        raise RuntimeError("Embeddings response missing data list")
    vectors: list[list[float]] = []
    for row in rows:
        embedding = row.get("embedding") if isinstance(row, dict) else None
        if not isinstance(embedding, list):
            continue
        vectors.append([float(item) for item in embedding])
    return model, vectors


def _embed_record_chunks(store: SQLiteRetrievalStore, record_id: int, env: ToolExecutionEnv) -> dict[str, Any]:
    chunks = store.chunk_texts_for_record(record_id)
    if not chunks:
        return {"enabled": False, "stored": 0}
    try:
        model, vectors = _embedding_vectors([str(chunk["text"]) for chunk in chunks], env)
    except RuntimeError as exc:
        return {"enabled": True, "stored": 0, "error": str(exc)}
    if not model or not vectors:
        return {"enabled": False, "stored": 0}
    stored = 0
    for chunk, vector in zip(chunks, vectors, strict=False):
        store.set_chunk_embedding(chunk_id=int(chunk["chunk_id"]), model=model, vector=vector)
        stored += 1
    return {"enabled": True, "stored": stored, "model": model}


def _searxng_base_url(env: ToolExecutionEnv) -> str:
    cfg = _search_cfg(env)
    base_url = str(cfg.get("searxng_base_url") or cfg.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        raise SearchError("SearXNG base URL not configured", failure_class=_FAILURE_CONFIG, recoverable=True)
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SearchError("SearXNG base URL must be an absolute http(s) URL", failure_class=_FAILURE_CONFIG, recoverable=True)
    return base_url


def _search_with_searxng(
    query: str,
    limit: int,
    env: ToolExecutionEnv,
    *,
    timeout_s: float = 20.0,
    retries: int = 1,
    retry_backoff_s: float = 0.5,
) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "format": "json", "language": "auto", "safesearch": "0"})
    req = _request(f"{_searxng_base_url(env)}/search?{params}", headers={"Accept": "application/json"})
    payload = _request_json(req, provider_name="SearXNG", timeout_s=timeout_s, retries=retries, retry_backoff_s=retry_backoff_s)
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("SearXNG response missing results list")
    out = _provider_payload(raw_results, limit, provider=SEARCH_PROVIDER_SEARXNG, provider_rank=0, query=query)
    out["query"] = query
    return out


def _tavily_api_key(env: ToolExecutionEnv) -> str:
    cfg = _search_cfg(env)
    env_name = str(cfg.get("tavily_api_key_env") or DEFAULT_TAVILY_API_KEY_ENV).strip()
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise SearchError(f"Tavily API key not configured: {env_name}", failure_class=_FAILURE_CONFIG, recoverable=True)
    return key


def _search_with_tavily(
    query: str,
    limit: int,
    env: ToolExecutionEnv,
    *,
    provider_rank: int = 0,
    timeout_s: float = 20.0,
    retries: int = 1,
    retry_backoff_s: float = 0.5,
) -> dict[str, Any]:
    key = _tavily_api_key(env)
    lowered = query.lower()
    is_newsish = any(token in lowered for token in ("latest", "today", "recent", "current", "news", "update"))
    req = _request(
        _TAVILY_ENDPOINT,
        data=json.dumps(
            {
                "query": query,
                "topic": "news" if is_newsish else "general",
                "search_depth": "advanced" if is_newsish else "basic",
                "max_results": limit,
                "include_answer": False,
                "include_raw_content": False,
            }
        ).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    payload = _request_json(req, provider_name="Tavily", timeout_s=timeout_s, retries=retries, retry_backoff_s=retry_backoff_s)
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("Tavily response missing results list")
    out = _provider_payload(raw_results, limit, provider=SEARCH_PROVIDER_TAVILY, provider_rank=provider_rank, query=query)
    out["query"] = query
    return out


def _normalize_result_item(item: dict, *, provider: str, provider_rank: int, query: str) -> dict[str, Any] | None:
    url = str(item.get("url", "")).strip()
    title = _clean_text(item.get("title", ""))
    if not url.startswith(("http://", "https://")) or not title:
        return None
    canonical_url = _canonicalize_url(url) or url
    domain = _host(canonical_url or url)
    snippet = _normalized_snippet(item)
    description = _raw_description(item)
    published_at = _raw_published_at(item)
    source_type = _source_type(domain, title, snippet or description)
    trust_score = round(_trust_score(domain, source_type), 2)
    query_match_score = _query_match_score(" ".join((title, snippet, description, domain)), query)
    freshness_score = _freshness_score(published_at)
    score = (trust_score * 0.5) + (query_match_score * 0.35) + (freshness_score * 0.1) + _ranking_bonus(source_type) + max(0.0, 0.05 - provider_rank * 0.02)
    return {
        "provider_rank": provider_rank,
        "title": title,
        "url": url,
        "canonical_url": canonical_url,
        "snippet": snippet,
        "description": description,
        "published_at": published_at,
        "domain": domain,
        "source_type": source_type,
        "trust_score": trust_score,
        "freshness_score": freshness_score,
        "query_match_score": query_match_score,
        "selection_reason": _selection_reason(source_type, query_match_score, freshness_score),
        "provider": provider,
        "_score": round(score, 4),
    }


def _provider_payload(raw_results: list[dict], limit: int, *, provider: str, provider_rank: int, query: str) -> dict[str, Any]:
    results = _normalize_results(raw_results, limit, provider=provider, provider_rank=provider_rank, query=query)
    if not results:
        raise SearchError(f"{provider.title()} returned no usable results", failure_class=_FAILURE_EMPTY_RESULTS, recoverable=True)
    return {"results": results, "provider": provider, "search_engine": provider}


def _classify_http_status(status: int) -> str:
    if status == 429:
        return _FAILURE_RATE_LIMIT
    if 500 <= status <= 599:
        return _FAILURE_NETWORK
    return _FAILURE_INVALID_RESPONSE


def _request_json(
    req: urllib.request.Request,
    *,
    provider_name: str,
    timeout_s: float,
    retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                content_type, body = _decode_response(resp)
        except urllib.error.HTTPError as exc:
            if should_retry(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise SearchError(
                f"{provider_name} returned HTTP {exc.code}",
                failure_class=_classify_http_status(int(exc.code)),
                recoverable=True,
            ) from exc
        except urllib.error.URLError as exc:
            if should_retry(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise SearchError(f"{provider_name} unreachable: {exc.reason}", failure_class=_FAILURE_NETWORK, recoverable=True) from exc

        if "json" not in content_type.lower():
            raise SearchError(f"{provider_name} returned a non-JSON response", failure_class=_FAILURE_INVALID_RESPONSE, recoverable=True)

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise SearchError(f"{provider_name} returned invalid JSON", failure_class=_FAILURE_INVALID_RESPONSE, recoverable=True) from exc


def _normalize_results(raw_results: list[dict], limit: int, *, provider: str, provider_rank: int, query: str) -> list[dict]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_result_item(item, provider=provider, provider_rank=provider_rank, query=query)
        if not normalized:
            continue
        key = str(normalized.get("canonical_url") or normalized.get("url"))
        existing = deduped.get(key)
        if existing is None or float(normalized["_score"]) > float(existing["_score"]):
            deduped[key] = normalized
    ranked = sorted(
        deduped.values(),
        key=lambda item: (-float(item["_score"]), -float(item["trust_score"]), str(item["domain"]), str(item["title"])),
    )
    trimmed = ranked[:limit]
    for index, item in enumerate(trimmed, start=1):
        item["rank"] = index
        item.pop("_score", None)
    return trimmed


def _cache_hit_to_result(hit: dict[str, Any], *, query: str, rank: int) -> dict[str, Any]:
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    source = str(hit.get("source") or metadata.get("url") or "").strip()
    canonical_url = _canonicalize_url(source) or source
    domain = str(metadata.get("domain") or _host(canonical_url)).strip()
    title = _clean_text(str(hit.get("title") or domain or source))
    snippet = _clean_text(str(hit.get("text") or ""))
    if len(snippet) > 500:
        snippet = snippet[:497].rstrip() + "..."
    source_type = str(metadata.get("source_type") or _source_type(domain, title, snippet))
    published_at = _parse_dateish(metadata.get("published_at"))
    query_match_score = _query_match_score(" ".join((title, snippet, domain)), query)
    return {
        "provider_rank": -1,
        "title": title,
        "url": source,
        "canonical_url": canonical_url,
        "snippet": snippet,
        "description": str(metadata.get("description") or ""),
        "published_at": published_at,
        "domain": domain,
        "source_type": source_type,
        "trust_score": round(_trust_score(domain, source_type), 2),
        "freshness_score": _freshness_score(published_at),
        "query_match_score": query_match_score,
        "selection_reason": "fresh cached source" if not bool(hit.get("stale")) else "stale cached source",
        "provider": "cache",
        "rank": rank,
        "cached": True,
        "stale": bool(hit.get("stale")),
        "record_id": int(hit.get("record_id", 0) or 0),
    }


def _cached_web_results(request: SearchRequest, env: ToolExecutionEnv) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _cache_first(env) or not _retrieval_enabled(env):
        return [], []
    try:
        hit_limit = max(request.limit, _min_usable_results(env)) * 4
        hits = _retrieval_store(env).search(request.query, top_k=hit_limit, sources=["web_page"])
    except Exception:
        return [], []
    results: list[dict[str, Any]] = []
    cache_hits: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for hit in hits:
        result = _cache_hit_to_result(hit, query=request.query, rank=len(results) + 1)
        if not result["url"]:
            continue
        source_key = str(result.get("canonical_url") or result.get("url") or result.get("record_id") or "").strip()
        if source_key and source_key in seen_sources:
            continue
        if source_key:
            seen_sources.add(source_key)
        results.append(result)
        cache_hits.append(
            {
                "record_id": result["record_id"],
                "source": result["url"],
                "title": result["title"],
                "stale": result["stale"],
            }
        )
        if len(results) >= request.limit:
            break
    return results, cache_hits


def _result_rank_score(item: dict[str, Any]) -> float:
    trust = float(item.get("trust_score", 0.0) or 0.0)
    query_match = float(item.get("query_match_score", 0.0) or 0.0)
    freshness = float(item.get("freshness_score", 0.0) or 0.0)
    source_bonus = _ranking_bonus(str(item.get("source_type", "")))
    provider_bonus = 0.08 if str(item.get("provider", "")) != "cache" else 0.0
    stale_penalty = 0.25 if bool(item.get("stale")) else 0.0
    return (trust * 0.5) + (query_match * 0.35) + (freshness * 0.1) + source_bonus + provider_bonus - stale_penalty


def _merge_search_results(cached_results: list[dict[str, Any]], provider_results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in [*cached_results, *provider_results]:
        key = str(item.get("canonical_url") or item.get("url") or "").strip()
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None or _result_rank_score(item) > _result_rank_score(existing):
            deduped[key] = dict(item)
    ranked = sorted(
        deduped.values(),
        key=lambda item: (
            -_result_rank_score(item),
            bool(item.get("stale")),
            str(item.get("provider", "")) == "cache",
            str(item.get("domain", "")),
            str(item.get("title", "")),
        ),
    )
    trimmed = ranked[:limit]
    for rank, item in enumerate(trimmed, start=1):
        item["rank"] = rank
    return trimmed


def _evidence_quality(results: list[dict[str, Any]], *, min_usable: int) -> str:
    if not results:
        return "none"
    fresh = [item for item in results if not bool(item.get("stale"))]
    high_signal = [
        item
        for item in fresh
        if float(item.get("trust_score", 0.0) or 0.0) >= 0.75 or str(item.get("source_type", "")) in {"official", "documentation", "reference"}
    ]
    if len(high_signal) >= min_usable:
        return "high"
    if len(fresh) >= min_usable:
        return "medium"
    return "low"


def _search_provider(provider: str, request: SearchRequest, env: ToolExecutionEnv, *, provider_rank: int) -> dict[str, Any]:
    timeout_s = _request_timeout_s(env)
    retries = _request_retries(env)
    retry_backoff_s = _request_retry_backoff_s(env)
    if provider == SEARCH_PROVIDER_SEARXNG:
        return _search_with_searxng(
            request.query,
            request.limit,
            env,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
    if provider == SEARCH_PROVIDER_TAVILY:
        return _search_with_tavily(
            request.query,
            request.limit,
            env,
            provider_rank=provider_rank,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
    raise SearchError("Only SearXNG and Tavily search providers are supported", failure_class=_FAILURE_UNSUPPORTED, recoverable=False)


def _search(query: str, limit: int, env: ToolExecutionEnv) -> dict[str, Any]:
    request = SearchRequest(query=query, limit=limit)
    min_usable = _min_usable_results(env)
    cached_results, cache_hits = _cached_web_results(request, env)
    if len([item for item in cached_results if not bool(item.get("stale"))]) >= min_usable:
        return SearchResponse(
            request=request,
            results=cached_results[:limit],
            cache_hits=cache_hits,
            provider="cache",
            degraded=False,
            evidence_quality=_evidence_quality(cached_results, min_usable=min_usable),
        ).to_payload()

    attempts: list[SearchAttempt] = []
    first_error: SearchError | None = None
    chain = _provider_chain(env)
    for provider_rank, provider in enumerate(chain):
        started = time.monotonic()
        try:
            payload = _search_provider(provider, request, env, provider_rank=provider_rank)
        except SearchError as exc:
            attempt = SearchAttempt(
                provider=provider,
                status="error",
                failure_class=exc.failure_class,
                error=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            attempts.append(attempt)
            if first_error is None:
                first_error = exc
            if not exc.recoverable:
                break
            continue
        except RuntimeError as exc:
            wrapped = SearchError(str(exc), failure_class=_FAILURE_INVALID_RESPONSE, recoverable=True)
            attempts.append(
                SearchAttempt(
                    provider=provider,
                    status="error",
                    failure_class=wrapped.failure_class,
                    error=str(wrapped),
                    latency_ms=int((time.monotonic() - started) * 1000),
                )
            )
            if first_error is None:
                first_error = wrapped
            continue

        provider_results = payload.get("results") if isinstance(payload, dict) else []
        results = provider_results if isinstance(provider_results, list) else []
        attempts.append(
            SearchAttempt(
                provider=provider,
                status="ok",
                latency_ms=int((time.monotonic() - started) * 1000),
                result_count=len(results),
            )
        )
        merged = _merge_search_results(cached_results, results, limit)
        if not merged:
            merged = [dict(item) for item in results[:limit]]
        quality = _evidence_quality(merged, min_usable=min_usable)
        return SearchResponse(
            request=request,
            results=merged,
            attempts=attempts,
            cache_hits=cache_hits,
            provider=str(payload.get("provider") or provider),
            degraded=bool(cache_hits) or len(attempts) > 1 or quality == "low",
            failure_class=attempts[0].failure_class if attempts and attempts[0].status == "error" else "",
            evidence_quality=quality,
        ).to_payload()

    if cached_results:
        quality = _evidence_quality(cached_results, min_usable=min_usable)
        return SearchResponse(
            request=request,
            results=cached_results[:limit],
            attempts=attempts,
            cache_hits=cache_hits,
            provider="cache",
            degraded=True,
            failure_class=attempts[-1].failure_class if attempts else "",
            evidence_quality=quality,
        ).to_payload()

    if len(chain) == 1 and first_error is not None:
        raise first_error
    errors = "; ".join(f"{attempt.provider}: {attempt.error}" for attempt in attempts if attempt.error)
    failure_class = attempts[-1].failure_class if attempts else _FAILURE_UNSUPPORTED
    raise SearchError(errors or "Search provider chain failed", failure_class=failure_class, recoverable=False)


def _is_private_or_local_host(host: str) -> bool:
    text = str(host or "").strip().lower().strip("[]")
    if not text:
        return True
    if text in _PRIVATE_HOST_LITERALS:
        return True
    if any(text.endswith(suffix) for suffix in _PRIVATE_HOST_SUFFIXES):
        return True

    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        addr = None
    if addr is not None:
        return bool(
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        )

    try:
        infos = socket.getaddrinfo(text, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False

    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_text = str(sockaddr[0]).strip().lower().strip("[]")
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False


def _meta_attr_map(raw_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw_attrs or ""):
        key = str(match.group(1) or "").strip().lower()
        value = match.group(3) or match.group(4) or match.group(5) or ""
        if key and key not in attrs:
            attrs[key] = unescape(value.strip())
    return attrs


def _meta_content(payload: str, keys: list[str]) -> str:
    wanted = {str(key).strip().lower() for key in keys if str(key).strip()}
    if not wanted:
        return ""
    for match in _META_RE.finditer(payload or ""):
        attrs = _meta_attr_map(match.group(1))
        probes = {attrs.get("name", "").lower(), attrs.get("property", "").lower(), attrs.get("itemprop", "").lower()}
        if wanted.intersection(probes):
            value = _clean_text(attrs.get("content", ""))
            if value:
                return value
    return ""


def _extract_headings(payload: str) -> list[str]:
    headings: list[str] = []
    for match in _HEADING_RE.finditer(payload or ""):
        text = _clean_text(_TAG_RE.sub(" ", match.group(2)))
        if text:
            headings.append(text)
    return headings[:8]


def _extract_blocks(payload: str) -> list[str]:
    blocks: list[str] = []
    body = _SCRIPT_STYLE_RE.sub(" ", payload or "")
    for match in _BLOCK_CAPTURE_RE.finditer(body):
        text = _clean_text(_TAG_RE.sub(" ", match.group(1)))
        if text:
            blocks.append(text)
    body = _BLOCK_TAG_RE.sub("\n", body)
    body = _TAG_RE.sub(" ", body)
    fallback_lines = [line for line in (_clean_text(chunk) for chunk in body.splitlines()) if line]
    merged: list[str] = []
    seen: set[str] = set()
    for item in blocks + fallback_lines:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _html_to_text(payload: str) -> str:
    blocks = _extract_blocks(payload)
    return "\n\n".join(blocks)


def _best_passages(text: str, limit: int = 3) -> list[str]:
    candidates = [chunk.strip() for chunk in re.split(r"\n{2,}", text) if chunk.strip()]
    if not candidates:
        candidates = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
    return candidates[:limit]


def _fetch(
    url: str,
    max_chars: int,
    *,
    timeout_s: float = 20.0,
    retries: int = 1,
    retry_backoff_s: float = 0.5,
    max_redirects: int = 5,
    min_chars: int = 20,
) -> dict[str, Any]:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    parsed_url = urllib.parse.urlparse(url)
    host = (parsed_url.hostname or "").strip()
    if _is_private_or_local_host(host):
        raise SearchError("Refusing to fetch private or local network URL", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False)

    attempt = 0
    redirect_count = 0
    current_url = url
    while True:
        try:
            with _open_no_redirect(_request(current_url), timeout_s=timeout_s) as resp:
                final_url = resp.geturl() or current_url
                content_type, payload = _decode_response(resp)
            break
        except urllib.error.HTTPError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            if status in _REDIRECT_HTTP_STATUS:
                location = ""
                if exc.headers is not None:
                    location = str(exc.headers.get("Location", "") or "").strip()
                if not location:
                    raise SearchError("Page fetch redirect missing Location header", failure_class=_FAILURE_INVALID_RESPONSE) from exc
                if redirect_count >= max_redirects:
                    raise SearchError("Page fetch exceeded redirect limit", failure_class=_FAILURE_INVALID_RESPONSE) from exc

                next_url = urllib.parse.urljoin(current_url, location)
                parsed_next = urllib.parse.urlparse(next_url)
                if parsed_next.scheme not in {"http", "https"}:
                    raise SearchError("Refusing to follow non-http redirect URL", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False) from exc
                next_host = (parsed_next.hostname or "").strip()
                if _is_private_or_local_host(next_host):
                    raise SearchError("Refusing to fetch private or local network URL", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False) from exc

                current_url = next_url
                redirect_count += 1
                attempt = 0
                continue

            if should_retry(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise SearchError(
                f"Page fetch returned HTTP {exc.code}",
                failure_class=_classify_http_status(int(exc.code)),
                recoverable=True,
            ) from exc
        except urllib.error.URLError as exc:
            if should_retry(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise SearchError(f"Page fetch failed: {exc.reason}", failure_class=_FAILURE_NETWORK, recoverable=True) from exc

    final_host = (urllib.parse.urlparse(final_url).hostname or "").strip()
    if _is_private_or_local_host(final_host):
        raise SearchError("Refusing to fetch private or local network URL", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False)

    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type and normalized_content_type not in _ALLOWED_FETCH_CONTENT_TYPES:
        raise SearchError(f"Unsupported content type: {normalized_content_type}", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False)

    title_match = _TITLE_RE.search(payload)
    title = _clean_text(title_match.group(1)) if title_match else ""
    description = ""
    published_at = ""
    author = ""
    headings: list[str] = []

    if "html" in normalized_content_type:
        description = _meta_content(payload, ["description", "og:description", "twitter:description"])
        published_at = _parse_dateish(_meta_content(payload, ["article:published_time", "og:published_time", "datePublished", "date"]))
        author = _meta_content(payload, ["author", "article:author"])
        headings = _extract_headings(payload)
        text = _html_to_text(payload)
    else:
        text = _clean_text(payload)

    truncated = len(text) > max_chars
    content = text[:max_chars].rstrip()
    best_passages = _best_passages(content)
    excerpt = best_passages[0] if best_passages else content[:280].strip()
    canonical_url = _canonicalize_url(final_url) or final_url
    domain = _host(canonical_url or final_url)
    source_type = _source_type(domain, title, description or excerpt)
    trust_score = round(_trust_score(domain, source_type), 2)
    extraction_quality = "high" if len(content) >= 1200 and headings else "medium" if len(content) >= 300 else "low"
    usable_text = len(content) >= min_chars and bool(best_passages)

    return {
        "url": url,
        "final_url": final_url,
        "canonical_url": canonical_url,
        "title": title,
        "description": description,
        "published_at": published_at,
        "author": author,
        "headings": headings,
        "content_type": content_type,
        "content": content,
        "excerpt": excerpt,
        "best_passages": best_passages,
        "truncated": truncated,
        "domain": domain,
        "source_type": source_type,
        "selection_reason": "fetched source content" + (" from a primary or official source" if source_type in {"official", "documentation"} else ""),
        "extraction_quality": extraction_quality,
        "content_chars": len(content),
        "usable_text": usable_text,
        "failure_class": "" if usable_text else _FAILURE_FETCH_UNUSABLE,
        "blocked_reason": "",
        "trust_score": trust_score,
        "fetched_at": int(time.time()),
    }


def _index_fetched_page(page: dict[str, Any], env: ToolExecutionEnv) -> dict[str, Any]:
    if not _retrieval_enabled(env):
        return {"indexed": False, "record_id": 0, "disabled": True}
    store = _retrieval_store(env)
    record = store.upsert_record(
        record_type="web_page",
        source=str(page.get("final_url") or page.get("url") or ""),
        canonical_source=str(page.get("canonical_url") or page.get("final_url") or page.get("url") or ""),
        title=str(page.get("title") or page.get("domain") or ""),
        text=str(page.get("content") or ""),
        fetched_at=int(page.get("fetched_at") or time.time()),
        ttl_seconds=_web_ttl_seconds(env),
        metadata={
            "url": page.get("url", ""),
            "domain": page.get("domain", ""),
            "source_type": page.get("source_type", ""),
            "published_at": page.get("published_at", ""),
            "author": page.get("author", ""),
            "description": page.get("description", ""),
        },
    )
    embedding = _embed_record_chunks(store, record.id, env) if record else {"enabled": False, "stored": 0}
    return {"indexed": bool(record), "record_id": record.id if record else 0, "embedding": embedding}


def _retrieve_knowledge(args: dict[str, Any], env: ToolExecutionEnv) -> dict[str, Any]:
    query = " ".join(str(args["query"]).split()).strip()
    if not query:
        raise ValueError("Retrieval query must not be empty")
    if not _retrieval_enabled(env):
        return {"query": query, "hits": [], "backend": "disabled"}
    raw_sources = args.get("sources") or []
    sources = [str(item).strip() for item in raw_sources if str(item).strip()] if isinstance(raw_sources, list) else []
    store = _retrieval_store(env)
    top_k = int(args.get("top_k", 5) or 5)
    hits = store.search(query, top_k=top_k, sources=sources)
    try:
        _model, vectors = _embedding_vectors([query], env)
    except RuntimeError:
        vectors = []
    if vectors:
        seen = {int(hit.get("chunk_id", 0) or 0) for hit in hits}
        for hit in store.dense_search(vectors[0], top_k=top_k, sources=sources):
            if int(hit.get("chunk_id", 0) or 0) not in seen:
                hits.append(hit)
                seen.add(int(hit.get("chunk_id", 0) or 0))
            if len(hits) >= top_k:
                break
    return {"query": query, "hits": hits[:top_k], "backend": "sqlite_hybrid" if vectors else "sqlite_fts"}


def _index_project(args: dict[str, Any], env: ToolExecutionEnv) -> dict[str, Any]:
    raw_paths = args.get("paths") or []
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("paths must contain at least one project file")
    if not _retrieval_enabled(env):
        return {"indexed": [], "count": 0, "disabled": True}
    limit = max(500, min(int(args.get("max_chars_per_file", 20000) or 20000), 100000))
    store = _retrieval_store(env)
    indexed: list[dict[str, Any]] = []
    for raw_path in raw_paths[:50]:
        path = str(raw_path).strip()
        if not path:
            continue
        content = env.project.read_file(path)
        record = store.upsert_record(
            record_type="project_document",
            source=path,
            canonical_source=path,
            title=path,
            text=content[:limit],
            metadata={"path": path, "truncated": len(content) > limit},
        )
        embedding = _embed_record_chunks(store, record.id, env) if record else {"enabled": False, "stored": 0}
        indexed.append({"path": path, "record_id": record.id if record else 0, "indexed": bool(record)})
        indexed[-1]["embedding"] = embedding
    return {"indexed": indexed, "count": sum(1 for item in indexed if item["indexed"])}


def _retrieval_stats(env: ToolExecutionEnv) -> dict[str, Any]:
    if not _retrieval_enabled(env):
        return {
            "path": str(configured_store_path(env.config if isinstance(env.config, dict) else {})),
            "records": 0,
            "chunks": 0,
            "stale_records": 0,
            "by_type": {},
            "backend": "disabled",
            "enabled": False,
            "embeddings": {"enabled": False, "mode": "openai_compatible", "ready": False},
        }
    stats = _retrieval_store(env).stats()
    retrieval_cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
    embeddings_cfg = retrieval_cfg.get("embeddings", {}) if isinstance(retrieval_cfg, dict) else {}
    stats["backend"] = "sqlite_fts"
    stats["embeddings"] = {
        "enabled": bool(embeddings_cfg.get("enabled", False)) if isinstance(embeddings_cfg, dict) else False,
        "mode": "openai_compatible",
        "ready": bool(embeddings_cfg.get("base_url") and embeddings_cfg.get("model")) if isinstance(embeddings_cfg, dict) else False,
    }
    return stats


def execute(tool_name: str, args: dict[str, Any], env: ToolExecutionEnv):
    if tool_name == "web_search":
        query = str(args["query"]).strip()
        limit = int(args.get("limit", 5) or 5)
        return _search(query, max(1, min(limit, 10)), env)
    if tool_name == "fetch_url":
        url = str(args["url"]).strip()
        max_chars = int(args.get("max_chars", 12000) or 12000)
        timeout_s = _request_timeout_s(env)
        retries = _request_retries(env)
        retry_backoff_s = _request_retry_backoff_s(env)
        max_redirects = _fetch_max_redirects(env)
        page = _fetch(
            url,
            max(500, min(max_chars, 30000)),
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            max_redirects=max_redirects,
            min_chars=_fetch_min_chars(env),
        )
        page["retrieval"] = _index_fetched_page(page, env)
        return page
    if tool_name == "retrieve_knowledge":
        return _retrieve_knowledge(args, env)
    if tool_name == "index_project":
        return _index_project(args, env)
    if tool_name == "retrieval_stats":
        return _retrieval_stats(env)
    if tool_name == "forget_retrieval_record":
        if not _retrieval_enabled(env):
            return {"deleted": False, "disabled": True}
        return {"deleted": _retrieval_store(env).forget(int(args["record_id"]))}
    raise ValueError(f"Unsupported tool: {tool_name}")
