import ipaddress
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from typing import Any

from core.retrieval import configured_store_path
from core.streaming import should_retry
from skills.runtime import ToolExecutionEnv

from . import search_engine as _search_engine

# fmt: off
TOOL_SPEC_ROWS = {
    "web_search": ("web_search", False, ("read", "check"), "Search the public web and return structured results with titles, URLs, snippets, and source metadata.", {"query": {"type": "string"}, "limit": {"type": "integer"}}, ("query",), False),
    "fetch_url": ("web_fetch", False, ("read",), "Fetch a URL and extract readable text content plus source metadata.", {"url": {"type": "string"}, "max_chars": {"type": "integer"}}, ("url",), False),
    "retrieve_knowledge": ("knowledge_retrieve", False, ("read", "check"), "Search the local SQLite retrieval index for web, memory, project, and tool outcome records.", {"query": {"type": "string"}, "top_k": {"type": "integer"}, "sources": {"type": "array", "items": {"type": "string"}}}, ("query",), False),
    "index_project": ("project_index", True, ("update",), "Index explicitly selected project files into the local retrieval store.", {"paths": {"type": "array", "items": {"type": "string"}}, "max_chars_per_file": {"type": "integer"}}, ("paths",), False),
    "retrieval_stats": ("retrieval_stats", False, ("check", "read"), "Return local retrieval database statistics and embedding availability.", {}, (), False),
    "forget_retrieval_record": ("retrieval_forget", True, ("delete", "remove"), "Delete a retrieval record by id.", {"record_id": {"type": "integer"}}, ("record_id",), False),
}
# fmt: on
_BLOCK_TAG_RE = re.compile(r"</?(?:p|div|section|article|li|ul|ol|h[1-6]|br|tr|td|th|blockquote)[^>]*>", re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_RE = re.compile(r"<meta\s+([^>]+)>", re.IGNORECASE)
_ATTR_RE = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*("([^"]*)"|\'([^\']*)\'|([^\s>]+))')
_HEADING_RE = re.compile(r"<h([1-3])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_CAPTURE_RE = re.compile(
    r"<(?:p|li|blockquote|h[1-6]|td|th)[^>]*>(.*?)</(?:p|li|blockquote|h[1-6]|td|th)>", re.IGNORECASE | re.DOTALL
)
_ALLOWED_FETCH_CONTENT_TYPES = ("text/html", "text/plain", "application/json", "application/xml", "text/xml")
_REDIRECT_HTTP_STATUS = {301, 302, 303, 307, 308}
_PRIVATE_HOST_SUFFIXES = (".local", ".internal", ".lan", ".home.arpa")
_PRIVATE_HOST_LITERALS = {"localhost", "localhost.localdomain", "0.0.0.0", "::", "::1"}
_MAX_FETCH_RESPONSE_BYTES = 2 * 1024 * 1024
_FAILURE_NETWORK = "network"
_FAILURE_INVALID_RESPONSE = "invalid_response"
_FAILURE_FETCH_BLOCKED = "fetch_blocked"
_FAILURE_FETCH_UNUSABLE = "fetch_unusable"


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
            addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved or addr.is_unspecified
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
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
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
            value = _search_engine._clean_text(attrs.get("content", ""))
            if value:
                return value
    return ""


def _extract_headings(payload: str) -> list[str]:
    headings: list[str] = []
    for match in _HEADING_RE.finditer(payload or ""):
        text = _search_engine._clean_text(_TAG_RE.sub(" ", match.group(2)))
        if text:
            headings.append(text)
    return headings[:8]


def _extract_blocks(payload: str) -> list[str]:
    blocks: list[str] = []
    body = _SCRIPT_STYLE_RE.sub(" ", payload or "")
    for match in _BLOCK_CAPTURE_RE.finditer(body):
        text = _search_engine._clean_text(_TAG_RE.sub(" ", match.group(1)))
        if text:
            blocks.append(text)
    body = _BLOCK_TAG_RE.sub("\n", body)
    body = _TAG_RE.sub(" ", body)
    fallback_lines = [line for line in (_search_engine._clean_text(chunk) for chunk in body.splitlines()) if line]
    merged: list[str] = []
    seen: set[str] = set()
    for item in blocks + fallback_lines:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


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
        raise _search_engine.SearchError(
            "Refusing to fetch private or local network URL", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False
        )

    attempt = 0
    redirect_count = 0
    current_url = url
    while True:
        try:
            with _search_engine._open_no_redirect(_search_engine._request(current_url), timeout_s=timeout_s) as resp:
                final_url = resp.geturl() or current_url
                content_type, payload = _search_engine._decode_response(resp, _MAX_FETCH_RESPONSE_BYTES)
            break
        except urllib.error.HTTPError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            if status in _REDIRECT_HTTP_STATUS:
                location = ""
                if exc.headers is not None:
                    location = str(exc.headers.get("Location", "") or "").strip()
                if not location:
                    raise _search_engine.SearchError(
                        "Page fetch redirect missing Location header", failure_class=_FAILURE_INVALID_RESPONSE
                    ) from exc
                if redirect_count >= max_redirects:
                    raise _search_engine.SearchError("Page fetch exceeded redirect limit", failure_class=_FAILURE_INVALID_RESPONSE) from exc

                next_url = urllib.parse.urljoin(current_url, location)
                parsed_next = urllib.parse.urlparse(next_url)
                if parsed_next.scheme not in {"http", "https"}:
                    raise _search_engine.SearchError(
                        "Refusing to follow non-http redirect URL", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False
                    ) from exc
                next_host = (parsed_next.hostname or "").strip()
                if _is_private_or_local_host(next_host):
                    raise _search_engine.SearchError(
                        "Refusing to fetch private or local network URL", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False
                    ) from exc

                current_url = next_url
                redirect_count += 1
                attempt = 0
                continue

            if should_retry(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise _search_engine.SearchError(
                f"Page fetch returned HTTP {exc.code}",
                failure_class=_search_engine._classify_http_status(int(exc.code)),
                recoverable=True,
            ) from exc
        except urllib.error.URLError as exc:
            if should_retry(exc) and attempt < retries:
                attempt += 1
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise _search_engine.SearchError(f"Page fetch failed: {exc.reason}", failure_class=_FAILURE_NETWORK, recoverable=True) from exc

    final_host = (urllib.parse.urlparse(final_url).hostname or "").strip()
    if _is_private_or_local_host(final_host):
        raise _search_engine.SearchError(
            "Refusing to fetch private or local network URL", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False
        )

    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type and normalized_content_type not in _ALLOWED_FETCH_CONTENT_TYPES:
        raise _search_engine.SearchError(
            f"Unsupported content type: {normalized_content_type}", failure_class=_FAILURE_FETCH_BLOCKED, recoverable=False
        )

    title_match = _TITLE_RE.search(payload)
    title = _search_engine._clean_text(title_match.group(1)) if title_match else ""
    description = ""
    published_at = ""
    author = ""
    headings: list[str] = []

    if "html" in normalized_content_type:
        description = _meta_content(payload, ["description", "og:description", "twitter:description"])
        published_at = _search_engine._parse_dateish(
            _meta_content(payload, ["article:published_time", "og:published_time", "datePublished", "date"])
        )
        author = _meta_content(payload, ["author", "article:author"])
        headings = _extract_headings(payload)
        text = "\n\n".join(_extract_blocks(payload))
    else:
        text = _search_engine._clean_text(payload)

    truncated = len(text) > max_chars
    content = text[:max_chars].rstrip()
    best_passages = _best_passages(content)
    excerpt = best_passages[0] if best_passages else content[:280].strip()
    canonical_url = _search_engine._canonicalize_url(final_url) or final_url
    domain = _search_engine._host(canonical_url or final_url)
    source_type = _search_engine._source_type(domain, title, description or excerpt)
    trust_score = round(_search_engine._trust_score(domain, source_type), 2)
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
        "selection_reason": "fetched source content"
        + (" from a primary or official source" if source_type in {"official", "documentation"} else ""),
        "extraction_quality": extraction_quality,
        "content_chars": len(content),
        "usable_text": usable_text,
        "failure_class": "" if usable_text else _FAILURE_FETCH_UNUSABLE,
        "blocked_reason": "",
        "trust_score": trust_score,
        "fetched_at": int(time.time()),
    }


def _index_fetched_page(page: dict[str, Any], env: ToolExecutionEnv) -> dict[str, Any]:
    if not _search_engine._retrieval_enabled(env):
        return {"indexed": False, "record_id": 0, "disabled": True}
    store = _search_engine._retrieval_store(env)
    record = store.upsert_record(
        record_type="web_page",
        source=str(page.get("final_url") or page.get("url") or ""),
        canonical_source=str(page.get("canonical_url") or page.get("final_url") or page.get("url") or ""),
        title=str(page.get("title") or page.get("domain") or ""),
        text=str(page.get("content") or ""),
        fetched_at=int(page.get("fetched_at") or time.time()),
        ttl_seconds=_search_engine._web_ttl_seconds(env),
        metadata={
            "url": page.get("url", ""),
            "domain": page.get("domain", ""),
            "source_type": page.get("source_type", ""),
            "published_at": page.get("published_at", ""),
            "author": page.get("author", ""),
            "description": page.get("description", ""),
        },
    )
    embedding = _search_engine._embed_record_chunks(store, record.id, env) if record else {"enabled": False, "stored": 0}
    return {"indexed": bool(record), "record_id": record.id if record else 0, "embedding": embedding}


def _retrieve_knowledge(args: dict[str, Any], env: ToolExecutionEnv) -> dict[str, Any]:
    query = " ".join(str(args["query"]).split()).strip()
    if not query:
        raise ValueError("Retrieval query must not be empty")
    if not _search_engine._retrieval_enabled(env):
        return {"query": query, "hits": [], "backend": "disabled"}
    raw_sources = args.get("sources") or []
    sources = [str(item).strip() for item in raw_sources if str(item).strip()] if isinstance(raw_sources, list) else []
    store = _search_engine._retrieval_store(env)
    top_k = int(args.get("top_k", 5) or 5)
    hits = store.search(query, top_k=top_k, sources=sources)
    try:
        model, vectors = _search_engine._embedding_vectors([query], env)
    except RuntimeError:
        vectors = []
    if vectors:
        cfg = env.config.get("retrieval", {}) if isinstance(env.config, dict) else {}
        dense_weight = float(cfg.get("dense_weight", 0.7)) if isinstance(cfg, dict) else 0.7
        lexical_weight = float(cfg.get("lexical_weight", 0.3)) if isinstance(cfg, dict) else 0.3
        hits = store.hybrid_search(
            query,
            vectors[0],
            model=model,
            top_k=top_k,
            sources=sources,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
        )
    return {"query": query, "hits": hits[:top_k], "backend": "sqlite_hybrid" if vectors else "sqlite_fts"}


def _index_project(args: dict[str, Any], env: ToolExecutionEnv) -> dict[str, Any]:
    raw_paths = args.get("paths") or []
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("paths must contain at least one project file")
    if not _search_engine._retrieval_enabled(env):
        return {"indexed": [], "count": 0, "disabled": True}
    limit = max(500, min(int(args.get("max_chars_per_file", 20000) or 20000), 100000))
    store = _search_engine._retrieval_store(env)
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
        embedding = _search_engine._embed_record_chunks(store, record.id, env) if record else {"enabled": False, "stored": 0}
        indexed.append({"path": path, "record_id": record.id if record else 0, "indexed": bool(record)})
        indexed[-1]["embedding"] = embedding
    return {"indexed": indexed, "count": sum(1 for item in indexed if item["indexed"])}


def _retrieval_stats(env: ToolExecutionEnv) -> dict[str, Any]:
    if not _search_engine._retrieval_enabled(env):
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
    stats = _search_engine._retrieval_store(env).stats()
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
        return _search_engine._search(query, max(1, min(limit, 10)), env)
    if tool_name == "fetch_url":
        url = str(args["url"]).strip()
        max_chars = int(args.get("max_chars", 12000) or 12000)
        timeout_s = _search_engine._request_timeout_s(env)
        retries = _search_engine._request_retries(env)
        retry_backoff_s = _search_engine._request_retry_backoff_s(env)
        max_redirects = _search_engine._fetch_max_redirects(env)
        page = _fetch(
            url,
            max(500, min(max_chars, 30000)),
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            max_redirects=max_redirects,
            min_chars=_search_engine._fetch_min_chars(env),
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
        if not _search_engine._retrieval_enabled(env):
            return {"deleted": False, "disabled": True}
        return {"deleted": _search_engine._retrieval_store(env).forget(int(args["record_id"]))}
    raise ValueError(f"Unsupported tool: {tool_name}")
