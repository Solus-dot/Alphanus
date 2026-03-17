from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict

from core.skills import ToolExecutionEnv

TOOL_SPECS = {
    "web_search": {
        "capability": "web_search",
        "description": "Search the public web and return structured results with titles, URLs, and snippets.",
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
        "description": "Fetch a URL and extract readable text content.",
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
_BLOCK_TAG_RE = re.compile(r"</?(?:p|div|section|article|li|ul|ol|h[1-6]|br|tr|td|th)[^>]*>", re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_ALLOWED_FETCH_CONTENT_TYPES = ("text/html", "text/plain", "application/json", "application/xml", "text/xml")
_TRUSTED_SOURCE_HINTS = (
    ".gov",
    ".edu",
    ".org",
    "wikipedia.org",
    "github.com",
    "official",
    "docs.",
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
    return re.sub(r"\s+", " ", text).strip()


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def _trust_score(host: str) -> float:
    if not host:
        return 0.0
    if any(host.endswith(suffix) for suffix in (".gov", ".mil", ".edu")):
        return 1.0
    if any(hint in host for hint in _TRUSTED_SOURCE_HINTS):
        return 0.85
    if host.count(".") >= 2:
        return 0.6
    return 0.45


def _provider_name(env: ToolExecutionEnv) -> str:
    search_cfg = env.config.get("search", {}) if isinstance(env.config, dict) else {}
    return str(search_cfg.get("provider", "tavily")).strip().lower() or "tavily"


def _api_key(env_var: str, missing_message: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(missing_message)
    return key


def _tavily_api_key() -> str:
    return _api_key("TAVILY_API_KEY", "Tavily API key not configured")


def _brave_api_key() -> str:
    return _api_key("BRAVE_SEARCH_API_KEY", "Brave Search API key not configured")


def _normalized_snippet(item: dict) -> str:
    for key in ("content", "description", "snippet"):
        value = _clean_text(str(item.get(key, "")).strip())
        if value:
            return value
    extra = item.get("extra_snippets")
    if isinstance(extra, list):
        for snippet in extra:
            value = _clean_text(str(snippet).strip())
            if value:
                return value
    return ""


def _provider_payload(raw_results: list[dict], limit: int, *, provider: str) -> Dict[str, Any]:
    results = _normalize_results(raw_results, limit)
    if not results:
        raise RuntimeError(f"{provider.title()} returned no usable results")
    return {"results": results, "provider": provider, "search_engine": provider}


def _normalize_results(raw_results: list[dict], limit: int) -> list[dict]:
    scored = []
    seen = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = _clean_text(str(item.get("title", "")).strip())
        snippet = _normalized_snippet(item)
        host = _host(url)
        if not url.startswith(("http://", "https://")) or not title or url in seen:
            continue
        seen.add(url)
        scored.append(
            (
                _trust_score(host),
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "domain": host,
                    "trust_score": round(_trust_score(host), 2),
                },
            )
        )
    scored.sort(key=lambda pair: (-pair[0], pair[1]["domain"], pair[1]["title"]))
    return [item for _, item in scored[:limit]]


def _search_with_tavily(query: str, limit: int) -> Dict[str, Any]:
    key = _tavily_api_key()
    payload = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
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

    payload = _provider_payload(raw_results, limit, provider="tavily")
    payload["query"] = query
    return payload


def _search_with_brave(query: str, limit: int) -> Dict[str, Any]:
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

    out = _provider_payload(raw_results, limit, provider="brave")
    out["query"] = query
    return out


def _search(query: str, limit: int, env: ToolExecutionEnv) -> Dict[str, Any]:
    provider = _provider_name(env)
    if provider == "tavily":
        return _search_with_tavily(query, limit)
    if provider == "brave":
        return _search_with_brave(query, limit)
    raise RuntimeError(f"Unsupported search provider: {provider}")


def _html_to_text(payload: str) -> str:
    body = _SCRIPT_STYLE_RE.sub(" ", payload)
    body = _BLOCK_TAG_RE.sub("\n", body)
    body = _TAG_RE.sub(" ", body)
    return _clean_text(body)


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

    if "html" in normalized_content_type:
        text = _html_to_text(payload)
    else:
        text = _clean_text(payload)

    truncated = len(text) > max_chars
    content = text[:max_chars].rstrip()
    return {
        "url": url,
        "final_url": final_url,
        "title": title,
        "content_type": content_type,
        "content": content,
        "truncated": truncated,
        "domain": _host(final_url),
        "trust_score": round(_trust_score(_host(final_url)), 2),
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
