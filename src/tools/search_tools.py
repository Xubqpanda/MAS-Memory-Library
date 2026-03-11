# src/tools/search_tools.py
from __future__ import annotations

import asyncio
import json
import os
import time
import warnings
from typing import Any

import requests

from .base import Tool


def read_page_jina(url: str) -> str:
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}",
        "X-Engine": "browser",
        "X-Return-Format": "markdown",
        "X-Retain-Images": "none",
        "X-Timeout": "10",
        "X-Token-Budget": "200000",
    }
    try:
        resp = requests.get(jina_url, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        return f"ReadPageError: {exc}"


def read_page_crawl4ai(url: str) -> str:
    try:
        from crawl4ai import AsyncWebCrawler
    except Exception:
        return "ReadPageError: crawl4ai not installed, set WEB_ACCESS_PROVIDER=jina or install crawl4ai."

    async def _crawl() -> str:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown or ""

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _crawl())
                text = future.result()
        else:
            text = loop.run_until_complete(_crawl())
    except RuntimeError:
        text = asyncio.run(_crawl())
    except Exception as exc:
        return f"ReadPageError: {exc}"

    return text if text else f"ReadPageError: empty content for {url}"


def read_page(url: str) -> str:
    provider = os.getenv("WEB_ACCESS_PROVIDER", "jina").lower()
    if provider == "crawl4ai":
        return read_page_crawl4ai(url)
    if provider not in {"jina", "crawl4ai"}:
        warnings.warn(
            f"Invalid WEB_ACCESS_PROVIDER='{provider}'. Using 'jina'.",
            UserWarning,
        )
    return read_page_jina(url)


def web_search_serper(query: str, serp_num: int = 5, max_retries: int = 3) -> tuple[list[dict[str, Any]], str]:
    if not query.strip():
        return [], "SearchError: empty query."

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return [], "SearchError: SERPER_API_KEY is not set."

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "location": "United States", "num": serp_num})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, data=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic", [])
            results: list[dict[str, Any]] = []
            for idx, page in enumerate(organic, 1):
                results.append(
                    {
                        "idx": idx,
                        "title": page.get("title", ""),
                        "link": page.get("link", ""),
                        "source": page.get("source", ""),
                        "date": page.get("date", ""),
                        "snippet": page.get("snippet", ""),
                    }
                )
            return results, ""
        except Exception as exc:
            if attempt == max_retries - 1:
                return [], f"SearchError: {exc}"
            time.sleep(2 ** attempt)
    return [], "SearchError: unexpected failure."


def web_search_searxng(
    query: str,
    base_url: str | None = None,
    serp_num: int = 5,
    max_retries: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    if not query.strip():
        return [], "SearchError: empty query."

    url_base = (base_url or os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8888")).rstrip("/")
    search_url = f"{url_base}/search"
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(search_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("results", [])
            if not raw_results:
                return [], f"No results found for '{query}' on SearXNG."

            results: list[dict[str, Any]] = []
            for idx, item in enumerate(raw_results[:serp_num], 1):
                engines = item.get("engines", ["Unknown"])
                results.append(
                    {
                        "idx": idx,
                        "title": item.get("title", "No title"),
                        "link": item.get("url", "#"),
                        "source": ", ".join(engines),
                        "date": item.get("publishedDate", ""),
                        "snippet": item.get("content", "No snippet"),
                    }
                )
            return results, ""
        except Exception as exc:
            if attempt == max_retries - 1:
                return [], f"SearchError(searxng): {exc}"
            time.sleep(2 ** attempt)
    return [], "SearchError(searxng): unexpected failure."


def web_search_google_custom(
    query: str,
    serp_num: int = 5,
    max_retries: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    if not query.strip():
        return [], "SearchError: empty query."

    api_key = os.getenv("GOOGLE_API_KEY")
    cx_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cx_id:
        return [], "SearchError: GOOGLE_API_KEY or GOOGLE_CSE_ID is not set."

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx_id, "q": query, "num": max(1, min(serp_num, 10))}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code in (403, 429):
                return [], f"SearchError(google): quota/rate-limited status={resp.status_code}"
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            if not items:
                return [], f"No results found for '{query}' on Google Custom Search."

            results: list[dict[str, Any]] = []
            for idx, item in enumerate(items, 1):
                pagemap = item.get("pagemap", {})
                metatags = pagemap.get("metatags", [{}])[0] if pagemap.get("metatags") else {}
                raw_date = metatags.get("article:published_time") or metatags.get("date") or ""
                results.append(
                    {
                        "idx": idx,
                        "title": item.get("title", "No title"),
                        "link": item.get("link", "#"),
                        "source": item.get("displayLink", "Google Search"),
                        "date": raw_date,
                        "snippet": item.get("snippet", "No snippet"),
                    }
                )
            return results, ""
        except Exception as exc:
            if attempt == max_retries - 1:
                return [], f"SearchError(google): {exc}"
            time.sleep(2 ** attempt)
    return [], "SearchError(google): unexpected failure."


def web_search_dispatch(
    query: str,
    provider: str | None = None,
    serp_num: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    engine = (provider or os.getenv("WEB_SEARCH_PROVIDER", "serper")).lower()
    if engine == "serper":
        return web_search_serper(query=query, serp_num=serp_num)
    if engine == "searxng":
        return web_search_searxng(query=query, serp_num=serp_num)
    if engine in {"google", "google_custom"}:
        return web_search_google_custom(query=query, serp_num=serp_num)
    return [], f"SearchError: unsupported provider '{engine}'. Use serper|searxng|google."


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search web using provider (serper/searxng/google)."

    def run(self, query: str, provider: str | None = None) -> str:
        results, err = web_search_dispatch(query=query, provider=provider, serp_num=5)
        if err:
            return err
        if not results:
            return "No results."
        lines = []
        for item in results:
            lines.append(
                f"{item['idx']}. [{item['title']}]({item['link']})\n"
                f"Source: {item['source']}  Date: {item['date'] or ''}\n"
                f"{item['snippet']}"
            )
        return "\n\n".join(lines)


class ReadPageTool(Tool):
    name = "read_page"
    description = "Read webpage markdown by URL using Jina or crawl4ai."

    def run(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "ReadPageError: URL must start with http:// or https://."
        return read_page(url)


class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Fetch Wikipedia summary for a query."

    def run(self, query: str) -> str:
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1,
            "inprop": "url",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            texts = []
            for page_id, page in pages.items():
                if int(page_id) < 0:
                    continue
                title = page.get("title", "Unknown")
                extract = (page.get("extract", "") or "").strip()
                url = page.get("fullurl", "")
                texts.append(f"[{title}]({url})\n{extract[:1000]}")
            return "\n\n".join(texts) if texts else f"No wiki results for: {query}"
        except Exception as exc:
            return f"WikiError: {exc}"


def build_default_search_tools() -> list[Tool]:
    return [WebSearchTool(), ReadPageTool(), WikiSearchTool()]
