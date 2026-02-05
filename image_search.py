from __future__ import annotations

import os
from urllib.parse import urlparse

import requests

from config import ALLOWED_IMAGE_DOMAINS, IMAGE_DOWNLOAD_TIMEOUT


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_allowed(url: str, allowed_domains: list[str] | None) -> bool:
    # If no domain restrictions, allow everything
    if not allowed_domains:
        return True
    host = _host(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in allowed_domains)


def search_images(query: str, max_results: int = 6, allowed_domains: list[str] | None = None) -> list[str]:
    """Return a list of image URLs. Uses DuckDuckGo if available.

    If allowed_domains is None or empty, searches the entire web without restrictions.
    """
    try:
        # Try new package name first
        from ddgs import DDGS
    except ImportError:
        try:
            # Fallback to old package name
            from duckduckgo_search import DDGS
        except Exception:
            return []

    # Only add site restrictions if domains are specified
    if allowed_domains:
        sites = " OR ".join([f"site:{d}" for d in allowed_domains])
        q = f"{query} {sites}"
    else:
        # No restrictions - search the entire web
        q = query

    urls: list[str] = []
    with DDGS() as ddgs:
        for r in ddgs.images(q, max_results=max_results):
            url = (r or {}).get("image")
            if not url:
                continue
            if _is_allowed(url, allowed_domains):
                urls.append(url)
    return urls


def search_web(query: str, max_results: int = 10) -> list[dict[str, str]]:
    """Search the web for text results using DuckDuckGo.

    Returns a list of search results with 'title', 'snippet', and 'url' keys.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except Exception:
            return []

    results: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if not r:
                    continue
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
    except Exception as e:
        print(f"Web search error: {e}")

    return results


def download_image(url: str, out_path: str) -> bool:
    try:
        r = requests.get(url, timeout=IMAGE_DOWNLOAD_TIMEOUT)
        if r.status_code != 200:
            return False
        ct = r.headers.get("content-type", "")
        if "image" not in ct:
            return False
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False
