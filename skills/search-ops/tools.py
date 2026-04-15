from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import ipaddress
from datetime import datetime, timezone
from html import unescape
from typing import Any, Dict, List

from core.skills import ToolExecutionEnv

TOOL_SPECS = {
    "web_search": {
        "capability": "web_search",
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
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_ALLOWED_FETCH_CONTENT_TYPES = ("text/html", "text/plain", "application/json", "application/xml", "text/xml")
_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}
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


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _request(url: str, *, data: bytes | None = None, headers: Dict[str, str] | None = None) -> urllib.request.Request:
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


def _query_tokens(query: str) -> List[str]:
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
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _freshness_score(published_at: str) -> float:
    if not published_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
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
    reasons: List[str] = []
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
    return str(search_cfg.get("provider", "tavily")).strip().lower() or "tavily"


def _search_cfg(env: ToolExecutionEnv) -> Dict[str, Any]:
    if not isinstance(getattr(env, "config", None), dict):
        return {}
    cfg = env.config.get("search")
    return cfg if isinstance(cfg, dict) else {}


def _cfg_float(search_cfg: Dict[str, Any], key: str, default: float, *, minimum: float = 0.0) -> float:
    raw = search_cfg.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _cfg_int(search_cfg: Dict[str, Any], key: str, default: int, *, minimum: int = 0) -> int:
    raw = search_cfg.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _cfg_bool(search_cfg: Dict[str, Any], key: str, default: bool) -> bool:
    raw = search_cfg.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(raw)


def _request_timeout_s(env: ToolExecutionEnv) -> float:
    return _cfg_float(_search_cfg(env), "request_timeout_s", 20.0, minimum=1.0)


def _request_retries(env: ToolExecutionEnv) -> int:
    return _cfg_int(_search_cfg(env), "request_retries", 1, minimum=0)


def _request_retry_backoff_s(env: ToolExecutionEnv) -> float:
    return _cfg_float(_search_cfg(env), "request_retry_backoff_s", 0.5, minimum=0.0)


def _fetch_max_redirects(env: ToolExecutionEnv) -> int:
    return _cfg_int(_search_cfg(env), "fetch_max_redirects", 5, minimum=0)


def _merge_providers_enabled(env: ToolExecutionEnv) -> bool:
    return _cfg_bool(_search_cfg(env), "merge_providers", True)


def _per_provider_limit(limit: int, env: ToolExecutionEnv) -> int:
    cfg = _search_cfg(env)
    requested = _cfg_int(cfg, "per_provider_limit", limit, minimum=1)
    return max(1, min(requested, 10))


def _provider_order(env: ToolExecutionEnv) -> List[str]:
    primary = _provider_name(env)
    if primary == "brave":
        return ["brave", "tavily"]
    return ["tavily", "brave"]


def _provider_configured(provider: str) -> bool:
    if provider == "tavily":
        return bool(os.environ.get("TAVILY_API_KEY", "").strip())
    if provider == "brave":
        return bool(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip())
    return False


def _api_key(env_var: str, missing_message: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(missing_message)
    return key


def _tavily_api_key() -> str:
    return _api_key("TAVILY_API_KEY", "Tavily API key not configured")


def _brave_api_key() -> str:
    return _api_key("BRAVE_SEARCH_API_KEY", "Brave Search API key not configured")


def _normalize_result_item(item: dict, *, provider: str, provider_rank: int, query: str) -> Dict[str, Any] | None:
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
        "provider": provider,
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


def _provider_payload(raw_results: list[dict], limit: int, *, provider: str, provider_rank: int, query: str) -> Dict[str, Any]:
    results = _normalize_results(raw_results, limit, provider=provider, provider_rank=provider_rank, query=query)
    if not results:
        raise RuntimeError(f"{provider.title()} returned no usable results")
    return {"results": results, "provider": provider, "search_engine": provider}


def _result_score(item: Dict[str, Any]) -> float:
    trust_score = float(item.get("trust_score", 0.0) or 0.0)
    query_match_score = float(item.get("query_match_score", 0.0) or 0.0)
    freshness_score = float(item.get("freshness_score", 0.0) or 0.0)
    provider_rank = int(item.get("provider_rank", 0) or 0)
    source_type = str(item.get("source_type", "") or "")
    return (
        (trust_score * 0.5)
        + (query_match_score * 0.35)
        + (freshness_score * 0.1)
        + _ranking_bonus(source_type)
        + max(0.0, 0.05 - provider_rank * 0.02)
    )


def _merge_search_payloads(payloads: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            record = dict(item)
            key = str(record.get("canonical_url") or record.get("url") or "").strip()
            if not key:
                continue
            score = _result_score(record)
            record["_score"] = score
            existing = deduped.get(key)
            if existing is None or score > float(existing.get("_score", -1.0)):
                deduped[key] = record

    ranked = sorted(
        deduped.values(),
        key=lambda item: (
            -float(item.get("_score", 0.0)),
            -float(item.get("trust_score", 0.0)),
            str(item.get("domain", "")),
            str(item.get("title", "")),
        ),
    )
    trimmed = ranked[:limit]
    for index, item in enumerate(trimmed, start=1):
        item["rank"] = index
        item.pop("_score", None)

    providers_used: List[str] = []
    for payload in payloads:
        provider = str(payload.get("provider", "")).strip().lower()
        if provider and provider not in providers_used:
            providers_used.append(provider)

    query = str(payloads[0].get("query", "")).strip() if payloads else ""
    return {
        "query": query,
        "provider": "multi",
        "search_engine": "multi",
        "providers_used": providers_used,
        "results": trimmed,
    }


def _retryable_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return int(getattr(exc, "code", 0) or 0) in _RETRYABLE_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", "")
        if isinstance(reason, (TimeoutError, ConnectionResetError)):
            return True
        text = str(reason).lower()
        return any(token in text for token in ("timed out", "temporarily", "reset"))
    if isinstance(exc, TimeoutError):
        return True
    return False


def _request_json(
    req: urllib.request.Request,
    *,
    provider_name: str,
    timeout_s: float,
    retries: int,
    retry_backoff_s: float,
) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                content_type, body = _decode_response(resp)
        except urllib.error.HTTPError as exc:
            if _retryable_error(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise RuntimeError(f"{provider_name} returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            if _retryable_error(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise RuntimeError(f"{provider_name} unreachable: {exc.reason}") from exc

        if "json" not in content_type.lower():
            raise RuntimeError(f"{provider_name} returned a non-JSON response")

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{provider_name} returned invalid JSON") from exc


def _normalize_results(raw_results: list[dict], limit: int, *, provider: str, provider_rank: int, query: str) -> list[dict]:
    deduped: Dict[str, Dict[str, Any]] = {}
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


def _search_with_tavily(
    query: str,
    limit: int,
    *,
    provider_rank: int = 0,
    timeout_s: float = 20.0,
    retries: int = 1,
    retry_backoff_s: float = 0.5,
) -> Dict[str, Any]:
    key = _tavily_api_key()
    lowered = query.lower()
    is_newsish = any(token in lowered for token in ("latest", "today", "recent", "current", "news", "update"))
    payload = {
        "query": query,
        "topic": "news" if is_newsish else "general",
        "search_depth": "advanced" if is_newsish else "basic",
        "max_results": limit,
        "include_answer": False,
        "include_raw_content": False,
    }
    req = _request(
        _TAVILY_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    payload = _request_json(
        req,
        provider_name="Tavily",
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("Tavily response missing results list")

    payload = _provider_payload(raw_results, limit, provider="tavily", provider_rank=provider_rank, query=query)
    payload["query"] = query
    return payload


def _search_with_brave(
    query: str,
    limit: int,
    *,
    provider_rank: int = 0,
    timeout_s: float = 20.0,
    retries: int = 1,
    retry_backoff_s: float = 0.5,
) -> Dict[str, Any]:
    key = _brave_api_key()
    params = urllib.parse.urlencode(
        {
            "q": query,
            "count": limit,
            "extra_snippets": "true",
            "text_decorations": "false",
            "spellcheck": "false",
        }
    )
    req = _request(
        f"{_BRAVE_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": key,
        },
    )
    payload = _request_json(
        req,
        provider_name="Brave",
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )

    web = payload.get("web", {})
    raw_results = web.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("Brave response missing web.results list")

    out = _provider_payload(raw_results, limit, provider="brave", provider_rank=provider_rank, query=query)
    out["query"] = query
    return out


def _search(query: str, limit: int, env: ToolExecutionEnv) -> Dict[str, Any]:
    errors: List[str] = []
    provider_chain: List[Dict[str, Any]] = []
    successful_payloads: List[Dict[str, Any]] = []
    merge_providers = _merge_providers_enabled(env)
    timeout_s = _request_timeout_s(env)
    retries = _request_retries(env)
    retry_backoff_s = _request_retry_backoff_s(env)
    provider_limit = _per_provider_limit(limit, env)

    for provider_rank, provider in enumerate(_provider_order(env)):
        if not _provider_configured(provider):
            if successful_payloads:
                continue
            message = (
                "Tavily API key not configured"
                if provider == "tavily"
                else "Brave Search API key not configured"
            )
            provider_chain.append({"provider": provider, "status": "error", "error": message})
            errors.append(message)
            continue

        try:
            payload = (
                _search_with_tavily(
                    query,
                    provider_limit,
                    provider_rank=provider_rank,
                    timeout_s=timeout_s,
                    retries=retries,
                    retry_backoff_s=retry_backoff_s,
                )
                if provider == "tavily"
                else _search_with_brave(
                    query,
                    provider_limit,
                    provider_rank=provider_rank,
                    timeout_s=timeout_s,
                    retries=retries,
                    retry_backoff_s=retry_backoff_s,
                )
            )
            provider_chain.append({"provider": provider, "status": "ok"})
            successful_payloads.append(payload)
            if not merge_providers:
                payload["provider_chain"] = provider_chain
                return payload
        except RuntimeError as exc:
            provider_chain.append({"provider": provider, "status": "error", "error": str(exc)})
            errors.append(str(exc))
            continue

    if successful_payloads:
        if len(successful_payloads) == 1:
            payload = successful_payloads[0]
            if len(payload.get("results", [])) > limit:
                payload["results"] = payload["results"][:limit]
                for index, item in enumerate(payload["results"], start=1):
                    item["rank"] = index
            payload["provider_chain"] = provider_chain
            return payload
        merged = _merge_search_payloads(successful_payloads, limit)
        merged["provider_chain"] = provider_chain
        return merged

    if errors:
        raise RuntimeError(errors[0])
    raise RuntimeError("No usable search provider available")


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


def _meta_attr_map(raw_attrs: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw_attrs or ""):
        key = str(match.group(1) or "").strip().lower()
        value = match.group(3) or match.group(4) or match.group(5) or ""
        if key and key not in attrs:
            attrs[key] = unescape(value.strip())
    return attrs


def _meta_content(payload: str, keys: List[str]) -> str:
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


def _extract_headings(payload: str) -> List[str]:
    headings: List[str] = []
    for match in _HEADING_RE.finditer(payload or ""):
        text = _clean_text(_TAG_RE.sub(" ", match.group(2)))
        if text:
            headings.append(text)
    return headings[:8]


def _extract_blocks(payload: str) -> List[str]:
    blocks: List[str] = []
    body = _SCRIPT_STYLE_RE.sub(" ", payload or "")
    for match in _BLOCK_CAPTURE_RE.finditer(body):
        text = _clean_text(_TAG_RE.sub(" ", match.group(1)))
        if text:
            blocks.append(text)
    body = _BLOCK_TAG_RE.sub("\n", body)
    body = _TAG_RE.sub(" ", body)
    fallback_lines = [line for line in (_clean_text(chunk) for chunk in body.splitlines()) if line]
    merged: List[str] = []
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


def _best_passages(text: str, limit: int = 3) -> List[str]:
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
) -> Dict[str, Any]:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    parsed_url = urllib.parse.urlparse(url)
    host = (parsed_url.hostname or "").strip()
    if _is_private_or_local_host(host):
        raise RuntimeError("Refusing to fetch private or local network URL")

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
                    raise RuntimeError("Page fetch redirect missing Location header") from exc
                if redirect_count >= max_redirects:
                    raise RuntimeError("Page fetch exceeded redirect limit") from exc

                next_url = urllib.parse.urljoin(current_url, location)
                parsed_next = urllib.parse.urlparse(next_url)
                if parsed_next.scheme not in {"http", "https"}:
                    raise RuntimeError("Refusing to follow non-http redirect URL") from exc
                next_host = (parsed_next.hostname or "").strip()
                if _is_private_or_local_host(next_host):
                    raise RuntimeError("Refusing to fetch private or local network URL") from exc

                current_url = next_url
                redirect_count += 1
                attempt = 0
                continue

            if _retryable_error(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise RuntimeError(f"Page fetch returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            if _retryable_error(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise RuntimeError(f"Page fetch failed: {exc.reason}") from exc

    final_host = (urllib.parse.urlparse(final_url).hostname or "").strip()
    if _is_private_or_local_host(final_host):
        raise RuntimeError("Refusing to fetch private or local network URL")

    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type and normalized_content_type not in _ALLOWED_FETCH_CONTENT_TYPES:
        raise RuntimeError(f"Unsupported content type: {normalized_content_type}")

    title_match = _TITLE_RE.search(payload)
    title = _clean_text(title_match.group(1)) if title_match else ""
    description = ""
    published_at = ""
    author = ""
    headings: List[str] = []

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
    usable_text = len(content) >= 20 and bool(best_passages)

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
        "trust_score": trust_score,
        "fetched_at": int(time.time()),
    }


def execute(tool_name: str, args: Dict[str, Any], env: ToolExecutionEnv):
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
        return _fetch(
            url,
            max(500, min(max_chars, 30000)),
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            max_redirects=max_redirects,
        )
    raise ValueError(f"Unsupported tool: {tool_name}")
