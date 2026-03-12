from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

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


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})


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


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._current_link: Optional[Dict[str, str]] = None
        self._current_snippet: List[str] = []
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = {key: value for key, value in attrs}
        class_names = set((attr_map.get("class") or "").split())
        if tag == "a" and "result__a" in class_names:
            self._capture_title = True
            self._current_link = {
                "href": attr_map.get("href", ""),
                "title": "",
            }
            return
        if "result__snippet" in class_names:
            self._capture_snippet = True
            self._current_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title and self._current_link is not None:
            self._capture_title = False
            href = _unwrap_duckduckgo_url(self._current_link["href"])
            title = _clean_text(self._current_link["title"])
            if href and title:
                self.results.append({"title": title, "url": href, "snippet": ""})
            self._current_link = None
            return
        if self._capture_snippet and tag in {"a", "div", "span"}:
            snippet = _clean_text("".join(self._current_snippet))
            if snippet and self.results and not self.results[-1]["snippet"]:
                self.results[-1]["snippet"] = snippet
            self._capture_snippet = False
            self._current_snippet = []

    def handle_data(self, data: str) -> None:
        if self._capture_title and self._current_link is not None:
            self._current_link["title"] += data
        elif self._capture_snippet:
            self._current_snippet.append(data)


def _unwrap_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/l/?") or url.startswith("https://duckduckgo.com/l/?") or url.startswith("http://duckduckgo.com/l/?"):
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        uddg = params.get("uddg", [""])[0]
        return urllib.parse.unquote(uddg) if uddg else ""
    return url if url.startswith("http://") or url.startswith("https://") else ""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def _search(query: str, limit: int) -> Dict[str, Any]:
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    try:
        with urllib.request.urlopen(_request(url), timeout=15) as resp:
            content_type, payload = _decode_response(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Search service returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Search service unreachable: {exc.reason}") from exc

    if "html" not in content_type.lower():
        raise RuntimeError("Search service returned a non-HTML response")

    parser = _DuckDuckGoParser()
    parser.feed(payload)

    unique: List[Dict[str, str]] = []
    seen = set()
    for result in parser.results:
        if result["url"] in seen:
            continue
        seen.add(result["url"])
        unique.append(
            {
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "domain": _host(result["url"]),
            }
        )
        if len(unique) >= limit:
            break

    return {"query": query, "results": unique, "search_engine": "duckduckgo"}


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
        return _search(query, max(1, min(limit, 10)))
    if tool_name == "fetch_url":
        url = str(args["url"]).strip()
        max_chars = int(args.get("max_chars", 12000) or 12000)
        return _fetch(url, max(500, min(max_chars, 30000)))
    raise ValueError(f"Unsupported tool: {tool_name}")
