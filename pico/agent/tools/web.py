"""Web tools: web_search and web_fetch."""

import json
import os
from typing import Any

import httpx
from loguru import logger

from pico.agent.tools.base import Tool, ToolResult
from pico.agent.tools.execution import ToolCapability, ToolEffect
from pico.security.network import validate_resolved_url, validate_url_target


class WebSearchTool(Tool):
    """Search the web using Serper."""

    capability = ToolCapability(effect=ToolEffect.EXTERNAL)
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str | None = None, max_results: int = 5, proxy: str | None = None):
        self._init_api_key = api_key
        self.max_results = max_results
        self.proxy = proxy

    @property
    def api_key(self) -> str:
        """Resolve API key at call time so env/config changes are picked up."""
        return self._init_api_key or os.environ.get("SERPER_API_KEY", "")

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if not self.api_key:
            return (
                "Error: Serper API key not configured. Set it in "
                "~/.pico/config.json under tools.web.search.apiKey "
                "(or export SERPER_API_KEY), then restart the gateway."
            )

        try:
            n = min(max(count or self.max_results, 1), 10)
            logger.debug("WebSearch: {}", "proxy enabled" if self.proxy else "direct connection")
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": n},
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "X-API-KEY": self.api_key,
                    },
                    timeout=10.0,
                )
                r.raise_for_status()

            data = r.json()
            results = data.get("organic", [])[:n]
            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            if answer := data.get("answerBox"):
                snippet = answer.get("answer") or answer.get("snippet")
                if snippet:
                    lines.append(f"Answer: {snippet}\n")
            if knowledge := data.get("knowledgeGraph"):
                title = knowledge.get("title")
                description = knowledge.get("description")
                if title or description:
                    lines.append(f"Knowledge: {title or ''}")
                    if description:
                        lines.append(f"   {description}")
            for i, item in enumerate(results, 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('link', '')}")
                if desc := item.get("snippet"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except httpx.ProxyError as e:
            logger.error("WebSearch proxy error: {}", e)
            return ToolResult(f"Proxy error: {e}", failed=True)
        except Exception as e:
            logger.error("WebSearch error: {}", e)
            return f"Error: {e}"


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Jina Reader."""

    capability = ToolCapability(effect=ToolEffect.EXTERNAL)
    name = "web_fetch"
    description = "Fetch URL and extract readable content via Jina Reader."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100},
        },
        "required": ["url"],
    }

    def __init__(self, api_key: str | None = None, max_chars: int = 50000, proxy: str | None = None):
        self._init_api_key = api_key
        self.max_chars = max_chars
        self.proxy = proxy

    @property
    def api_key(self) -> str:
        """Resolve API key at call time so env/config changes are picked up."""
        return self._init_api_key or os.environ.get("JINA_API_KEY", "")

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:  # noqa: N803  （LLM 工具模式使用驼峰命名）
        max_chars = maxChars or self.max_chars
        is_valid, error_msg = validate_url_target(url)
        if not is_valid:
            return ToolResult(
                json.dumps(
                    {"error": f"URL validation failed: {error_msg}", "url": url},
                    ensure_ascii=False,
                ),
                failed=True,
            )

        try:
            logger.debug("WebFetch: {}", "proxy enabled" if self.proxy else "direct connection")
            # Reader 当前的 X-Base 合约在 JSON data.url 中返回快照链接。Reader 已完成抓取，
            # Pico 只将验证用作返回内容的门禁。
            headers = {"Accept": "application/json", "X-Base": "final"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            async with httpx.AsyncClient(timeout=30.0, proxy=self.proxy) as client:
                r = await client.get(f"https://r.jina.ai/{url}", headers=headers)
                r.raise_for_status()

            payload = r.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                return ToolResult(
                    json.dumps(
                        {"error": "Invalid Jina Reader response: missing data object", "url": url},
                        ensure_ascii=False,
                    ),
                    failed=True,
                )

            final_url = data.get("url")
            if not isinstance(final_url, str) or not final_url:
                return ToolResult(
                    json.dumps(
                        {"error": "Invalid Jina Reader response: missing final URL", "url": url},
                        ensure_ascii=False,
                    ),
                    failed=True,
                )

            is_valid, error_msg = validate_resolved_url(final_url)
            if not is_valid:
                return ToolResult(
                    json.dumps(
                        {"error": f"Final URL validation failed: {error_msg}", "url": url},
                        ensure_ascii=False,
                    ),
                    failed=True,
                )

            text = data.get("content")
            if not isinstance(text, str):
                return ToolResult(
                    json.dumps(
                        {"error": "Invalid Jina Reader response: missing content", "url": url},
                        ensure_ascii=False,
                    ),
                    failed=True,
                )

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps(
                {
                    "url": url,
                    "finalUrl": final_url,
                    "status": r.status_code,
                    "extractor": "jina-reader",
                    "extractMode": extractMode,
                    "truncated": truncated,
                    "length": len(text),
                    "text": text,
                },
                ensure_ascii=False,
            )
        except httpx.ProxyError as e:
            logger.error("WebFetch proxy error for {}: {}", url, e)
            return ToolResult(
                json.dumps({"error": f"Proxy error: {e}", "url": url}, ensure_ascii=False),
                failed=True,
            )
        except Exception as e:
            logger.error("WebFetch error for {}: {}", url, e)
            return ToolResult(
                json.dumps({"error": str(e), "url": url}, ensure_ascii=False),
                failed=True,
            )
