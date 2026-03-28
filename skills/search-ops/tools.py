from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
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


def _request(url: str, *, data: bytes | None = None, headers: Dict[str, str] | None = None) -> urllib.request.Request:
    merged = {"User-Agent": _USER_AGENT}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, data=data, headers=merged)


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


def _provider_order(env: ToolExecutionEnv) -> List[str]:
    primary = _provider_name(env)
    if primary == "brave":
        return ["brave", "tavily"]
    return ["tavily", "brave"]


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
        "_score": round(score, 4),
    }


def _provider_payload(raw_results: list[dict], limit: int, *, provider: str, provider_rank: int, query: str) -> Dict[str, Any]:
    results = _normalize_results(raw_results, limit, provider=provider, provider_rank=provider_rank, query=query)
    if not results:
        raise RuntimeError(f"{provider.title()} returned no usable results")
    return {"results": results, "provider": provider, "search_engine": provider}


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


def _search_with_tavily(query: str, limit: int, *, provider_rank: int = 0) -> Dict[str, Any]:
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
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type, body = _decode_response(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Tavily returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Tavily unreachable: {exc.reason}") from exc

    if "json" not in content_type.lower():
        raise RuntimeError("Tavily returned a non-JSON response")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Tavily returned invalid JSON") from exc

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("Tavily response missing results list")

    payload = _provider_payload(raw_results, limit, provider="tavily", provider_rank=provider_rank, query=query)
    payload["query"] = query
    return payload


def _search_with_brave(query: str, limit: int, *, provider_rank: int = 0) -> Dict[str, Any]:
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
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type, body = _decode_response(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Brave returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Brave unreachable: {exc.reason}") from exc

    if "json" not in content_type.lower():
        raise RuntimeError("Brave returned a non-JSON response")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Brave returned invalid JSON") from exc

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
    for provider_rank, provider in enumerate(_provider_order(env)):
        try:
            payload = _search_with_tavily(query, limit, provider_rank=provider_rank) if provider == "tavily" else _search_with_brave(query, limit, provider_rank=provider_rank)
            payload["provider_chain"] = provider_chain + [{"provider": provider, "status": "ok"}]
            return payload
        except RuntimeError as exc:
            provider_chain.append({"provider": provider, "status": "error", "error": str(exc)})
            errors.append(str(exc))
            continue
    if errors:
        raise RuntimeError(errors[0])
    raise RuntimeError("No usable search provider available")


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


def _fetch(url: str, max_chars: int) -> Dict[str, Any]:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    try:
        with urllib.request.urlopen(_request(url), timeout=20) as resp:
            final_url = resp.geturl()
            content_type, payload = _decode_response(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Page fetch returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Page fetch failed: {exc.reason}") from exc

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
        return _fetch(url, max(500, min(max_chars, 30000)))
    raise ValueError(f"Unsupported tool: {tool_name}")
