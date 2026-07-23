import json
import os
import re
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

_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_TRUSTED_SOURCE_HINTS = (".gov", ".mil", ".edu", ".org", "wikipedia.org", "github.com", "official", "docs.")
_TRACKING_QUERY_KEYS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
_MAX_JSON_RESPONSE_BYTES = 4 * 1024 * 1024
_DATE_FORMATS = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")
_FAILURE_CONFIG = "config"
_FAILURE_NETWORK = "network"
_FAILURE_RATE_LIMIT = "rate_limit"
_FAILURE_INVALID_RESPONSE = "invalid_response"
_FAILURE_EMPTY_RESULTS = "empty_results"
_FAILURE_UNSUPPORTED = "unsupported_provider"


class SearchError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str, recoverable: bool = True) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.recoverable = recoverable


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    limit: int


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
        payload = asdict(self)
        payload["query"] = self.request.query
        payload["provider"] = payload["search_engine"] = provider
        payload.pop("request", None)
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


def _decode_response(resp, max_bytes: int) -> tuple[str, str]:
    content_length = resp.headers.get("Content-Length", "")
    if content_length.isdigit() and int(content_length) > max_bytes:
        raise SearchError("HTTP response exceeds byte limit", failure_class=_FAILURE_INVALID_RESPONSE, recoverable=True)
    raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise SearchError("HTTP response exceeds byte limit", failure_class=_FAILURE_INVALID_RESPONSE, recoverable=True)
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
    return (
        0.85
        if any(hint in host for hint in _TRUSTED_SOURCE_HINTS)
        else 0.75
        if source_type == "news"
        else 0.6
        if host.count(".") >= 2
        else 0.45
    )


def _normalized_snippet(item: dict) -> str:
    extra = item.get("extra_snippets")
    candidates = [item.get(key, "") for key in ("content", "description", "snippet")]
    candidates.extend(extra if isinstance(extra, list) else [])
    return next((value for candidate in candidates if (value := _clean_text(candidate))), "")


def _raw_published_at(item: dict) -> str:
    return next(
        (parsed for key in ("published_date", "published_at", "date", "last_updated") if (parsed := _parse_dateish(item.get(key)))),
        "",
    )


def _selection_reason(source_type: str, query_match_score: float, freshness_score: float) -> str:
    source_reason = {"official": "primary or official source", "documentation": "primary or official source", "news": "news reporting"}
    reasons = [source_reason[source_type]] if source_type in source_reason else []
    if query_match_score >= 0.7:
        reasons.append("strong query match")
    elif query_match_score >= 0.4:
        reasons.append("relevant query match")
    if freshness_score >= 0.7:
        reasons.append("recent metadata")
    return ", ".join(reasons[:3]) or "best available result"


_RANKING_BONUS = {"official": 0.18, "documentation": 0.18, "reference": 0.08, "news": 0.04}


def _search_cfg(env: ToolExecutionEnv) -> dict[str, Any]:
    if not isinstance(getattr(env, "config", None), dict):
        return {}
    cfg = env.config.get("search")
    return cfg if isinstance(cfg, dict) else {}


def _cfg_number(search_cfg: dict[str, Any], key: str, default: int | float, *, minimum: int | float = 0) -> int | float:
    try:
        value = type(default)(search_cfg.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _request_timeout_s(env: ToolExecutionEnv) -> float:
    return float(_cfg_number(_search_cfg(env), "request_timeout_s", 20.0, minimum=1.0))


def _request_retries(env: ToolExecutionEnv) -> int:
    return int(_cfg_number(_search_cfg(env), "request_retries", 1))


def _request_retry_backoff_s(env: ToolExecutionEnv) -> float:
    return float(_cfg_number(_search_cfg(env), "request_retry_backoff_s", 0.5))


def _fetch_max_redirects(env: ToolExecutionEnv) -> int:
    return int(_cfg_number(_search_cfg(env), "fetch_max_redirects", 5))


def _min_usable_results(env: ToolExecutionEnv) -> int:
    return int(_cfg_number(_search_cfg(env), "min_usable_results", 1, minimum=1))


def _fetch_min_chars(env: ToolExecutionEnv) -> int:
    return int(_cfg_number(_search_cfg(env), "fetch_min_chars", 20, minimum=1))


def _provider_chain(env: ToolExecutionEnv) -> list[str]:
    cfg = _search_cfg(env)
    raw_chain = cfg.get("provider_chain")
    chain: list[str] = []
    if isinstance(raw_chain, list):
        chain = [str(item).strip().lower() for item in raw_chain if str(item).strip()]
    elif isinstance(raw_chain, str) and raw_chain.strip():
        chain = [item.strip().lower() for item in raw_chain.split(",") if item.strip()]
    if not chain:
        provider = str(cfg.get("provider", SEARCH_PROVIDER_SEARXNG)).strip().lower() or SEARCH_PROVIDER_SEARXNG
        fallback = str(cfg.get("fallback_provider", "")).strip().lower()
        chain = [provider]
        if fallback and fallback != provider:
            chain.append(fallback)
    out: list[str] = []
    for provider in chain:
        if provider != "none" and provider not in out:
            out.append(provider)
    return out or [SEARCH_PROVIDER_SEARXNG]


def _web_ttl_seconds(env: ToolExecutionEnv) -> int:
    cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
    hours = float(_cfg_number(cfg, "web_ttl_hours", 72.0)) if isinstance(cfg, dict) else 72.0
    return max(0, int(hours * 3600))


def _retrieval_enabled(env: ToolExecutionEnv) -> bool:
    cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
    return coerce_bool(cfg.get("enabled", True), True) if isinstance(cfg, dict) else True


def _retrieval_store(env: ToolExecutionEnv) -> SQLiteRetrievalStore:
    cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
    limit = int(cfg.get("candidate_limit", 2000)) if isinstance(cfg, dict) else 2000
    return SQLiteRetrievalStore(configured_store_path(env.config if isinstance(env.config, dict) else {}), candidate_limit=limit)


def _embedding_cfg(env: ToolExecutionEnv) -> dict[str, Any]:
    retrieval_cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
    embeddings_cfg = retrieval_cfg.get("embeddings", {}) if isinstance(retrieval_cfg, dict) else {}
    return embeddings_cfg if isinstance(embeddings_cfg, dict) else {}


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
    vectors: list[list[float]] = []
    expected_dimensions = max(0, int(cfg.get("dimensions", 0) or 0))
    batch_size = max(1, min(int(cfg.get("batch_size", 32) or 32), 256))
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        trimmed_url = base_url.rstrip("/")
        req = _request(
            trimmed_url if trimmed_url.endswith("/embeddings") else f"{trimmed_url}{OPENAI_EMBEDDINGS_PATH}",
            data=json.dumps({"model": model, "input": batch}).encode("utf-8"),
            headers=headers,
        )
        response = _request_json(req, provider_name="Embeddings", timeout_s=_request_timeout_s(env), retries=0, retry_backoff_s=0.0)
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise RuntimeError("Embeddings response count does not match the input batch")
        for row in rows:
            embedding = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError("Embeddings response contains an invalid vector")
            vector = [float(item) for item in embedding]
            if expected_dimensions and len(vector) != expected_dimensions:
                raise RuntimeError(f"Embedding dimension mismatch: expected {expected_dimensions}, got {len(vector)}")
            vectors.append(vector)
    return model, vectors


def _embed_record_chunks(store: SQLiteRetrievalStore, record_id: int, env: ToolExecutionEnv) -> dict[str, Any]:
    chunks = store.chunk_texts_for_record(record_id)
    if not chunks:
        return {"enabled": False, "stored": 0}
    cfg = _embedding_cfg(env)
    model = str(cfg.get("model") or "").strip()
    dimensions = max(0, int(cfg.get("dimensions", 0) or 0))
    cached = [store.cached_embedding(str(chunk["text"]), model=model, dimensions=dimensions) for chunk in chunks] if model else []
    missing = [str(chunk["text"]) for chunk, vector in zip(chunks, cached, strict=False) if vector is None]
    try:
        embedded_model, generated = _embedding_vectors(missing, env)
    except RuntimeError as exc:
        return {"enabled": True, "stored": 0, "error": str(exc)}
    if not model:
        model = embedded_model
    if not model or (not generated and not any(cached)):
        return {"enabled": False, "stored": 0}
    generated_iter = iter(generated)
    vectors = [vector if vector is not None else next(generated_iter) for vector in cached]
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
    description = next((value for key in ("description", "content", "snippet") if (value := _clean_text(item.get(key, "")))), "")
    published_at = _raw_published_at(item)
    source_type = _source_type(domain, title, snippet or description)
    trust_score = round(_trust_score(domain, source_type), 2)
    query_match_score = _query_match_score(" ".join((title, snippet, description, domain)), query)
    freshness_score = _freshness_score(published_at)
    score = (
        (trust_score * 0.5)
        + (query_match_score * 0.35)
        + (freshness_score * 0.1)
        + _RANKING_BONUS.get(source_type, 0.0)
        + max(0.0, 0.05 - provider_rank * 0.02)
    )
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
    return _FAILURE_RATE_LIMIT if status == 429 else _FAILURE_NETWORK if 500 <= status <= 599 else _FAILURE_INVALID_RESPONSE


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
                content_type, body = _decode_response(resp, _MAX_JSON_RESPONSE_BYTES)
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
    if not coerce_bool(_search_cfg(env).get("cache_first", True), True) or not _retrieval_enabled(env):
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
    source_bonus = _RANKING_BONUS.get(str(item.get("source_type", "")), 0.0)
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
        if float(item.get("trust_score", 0.0) or 0.0) >= 0.75
        or str(item.get("source_type", "")) in {"official", "documentation", "reference"}
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
        except (RuntimeError, SearchError) as raw_error:
            error = (
                raw_error
                if isinstance(raw_error, SearchError)
                else SearchError(str(raw_error), failure_class=_FAILURE_INVALID_RESPONSE, recoverable=True)
            )
            attempts.append(
                SearchAttempt(
                    provider=provider,
                    status="error",
                    failure_class=error.failure_class,
                    error=str(error),
                    latency_ms=int((time.monotonic() - started) * 1000),
                )
            )
            if first_error is None:
                first_error = error
            if not error.recoverable:
                break
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
