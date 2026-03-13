from __future__ import annotations

import json
import os
import re
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


def _tavily_api_key(env: ToolExecutionEnv) -> str:
    search_cfg = env.config.get("search", {}) if isinstance(env.config, dict) else {}
    key = str(search_cfg.get("tavily_api_key", "")).strip() or os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Tavily API key not configured")
    return key


def _search(query: str, limit: int, env: ToolExecutionEnv) -> Dict[str, Any]:
    key = _tavily_api_key(env)
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

    results = []
    seen = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = _clean_text(str(item.get("title", "")).strip())
        snippet = _clean_text(str(item.get("content", "")).strip())
        if not url.startswith(("http://", "https://")) or not title or url in seen:
            continue
        seen.add(url)
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": _host(url),
            }
        )
        if len(results) >= limit:
            break

    if not results:
        raise RuntimeError("Tavily returned no usable results")

    return {"query": query, "results": results, "search_engine": "tavily"}


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

    title_match = _TITLE_RE.search(payload)
    title = _clean_text(title_match.group(1)) if title_match else ""

    if "html" in content_type.lower():
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
