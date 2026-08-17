"""LiteLLM provider implementation for multi-provider support."""

import hashlib
import os
import secrets
import string
import warnings
from collections.abc import AsyncIterator
from typing import Any

import json_repair
from loguru import logger

from pico.providers.base import LLMProvider, LLMResponse, StreamDelta, ToolCallRequest
from pico.providers.litellm_setup import import_litellm
from pico.providers.registry import find_by_model, find_gateway

litellm = import_litellm()
acompletion = litellm.acompletion

# LiteLLM 的异步日志 worker（LoggingWorker）会把队列绑定到单一 event loop。
# Pico 每个 Turn 使用新 loop（每次调用 asyncio.run），所以下一个 Turn 会重置
# 队列，所有待处理的 ``Logging.async_*_handler`` 协程都会在未 await 的情况下
# 被丢弃。Python 随后打印 ``coroutine ... was never awaited`` RuntimeWarning，
# 并污染 Ink TUI 渲染。被丢弃的是 LiteLLM 自身的成功/失败日志回调，Pico 不依赖
# 它们。过滤范围只限于 LiteLLM 的 ``Logging`` handler；宽泛的
# ``coroutine '.*'`` 模式也会掩盖 Pico 自身真正的未 await 错误。
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'Logging\.async_.*' was never awaited",
    category=RuntimeWarning,
)

# 标准 chat-completion 消息字段。
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"})
_ANTHROPIC_EXTRA_KEYS = frozenset({"thinking_blocks"})
_ALNUM = string.ascii_letters + string.digits

# LiteLLM 默认给 OpenRouter 请求设置 X-Title="liteLLM" 和
# HTTP-Referer="https://litellm.ai"，这会让 openrouter.ai/apps 把流量归因给
# LiteLLM 而非 Pico。这里显式覆盖默认值；用户提供的 extra_headers 优先级更高。
_OPENROUTER_ATTRIBUTION: dict[str, str] = {
    "HTTP-Referer": "https://gitee.com/htxoffical/pico-harness",
    "X-Title": "Pico Agent Harness",
    "X-OpenRouter-Title": "Pico Agent Harness",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _usage_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_usage_payload(usage: Any) -> dict[str, Any]:
    normalized = {
        name: value
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if (value := _usage_field(usage, name)) is not None
    }
    prompt_details = _usage_field(usage, "prompt_tokens_details")
    completion_details = _usage_field(usage, "completion_tokens_details")
    cache_read = _first_present(
        _usage_field(usage, "prompt_cache_hit_tokens"),
        _usage_field(usage, "cache_read_input_tokens"),
        _usage_field(prompt_details, "cached_tokens"),
        _usage_field(usage, "_cache_read_input_tokens"),
    )
    cache_write = _first_present(
        _usage_field(usage, "cache_creation_input_tokens"),
        _usage_field(prompt_details, "cache_write_tokens"),
        _usage_field(prompt_details, "cache_creation_tokens"),
        _usage_field(usage, "_cache_creation_input_tokens"),
    )
    cache_miss = _usage_field(usage, "prompt_cache_miss_tokens")
    reasoning = _first_present(
        _usage_field(usage, "reasoning_tokens"),
        _usage_field(completion_details, "reasoning_tokens"),
    )
    for key, value in (
        ("cache_read_input_tokens", cache_read),
        ("cache_creation_input_tokens", cache_write),
        ("cache_miss_input_tokens", cache_miss),
        ("reasoning_tokens", reasoning),
    ):
        if value is not None:
            normalized[key] = value
    return normalized


def _short_tool_id() -> str:
    """Generate a 9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.

    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
        disable_auto_cache_control: bool = True,
        extra_body: dict[str, Any] | None = None,
        transport_num_retries: int | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        # CallEfficiency 负责在 Runtime 接缝显式重写请求。
        # False 仅为历史实验兼容而保留。
        self.disable_auto_cache_control = disable_auto_cache_control
        # Provider 专属请求体扩展会原样转发给 LiteLLM。常见用途是固定
        # OpenRouter 路由 affinity，以维持 prompt cache 热度，
        #   参数示例：extra_body={"provider": {"order": ["Anthropic"], "allow_fallbacks": False}}
        self.extra_body = extra_body or {}
        self.set_transport_num_retries(transport_num_retries)

        # 检测网关/本地部署。来自配置键的 provider_name 是主要信号；
        # api_key / api_base 仅用于自动检测回退。
        self._gateway = find_gateway(provider_name, api_key, api_base)
        if self._gateway and self._gateway.name == "openrouter":
            self.extra_headers = {**_OPENROUTER_ATTRIBUTION, **self.extra_headers}

        # 配置环境变量。
        if api_key:
            self._setup_env(api_key, api_base, default_model)

        if api_base:
            litellm.api_base = api_base

        # 移除 Provider 不支持的参数，例如 gpt-5 会拒绝部分参数。
        litellm.drop_params = True

    def set_transport_num_retries(
        self,
        num_retries: int | None,
    ) -> None:
        if num_retries is not None and (
            isinstance(num_retries, bool) or not isinstance(num_retries, int) or num_retries < 0
        ):
            raise ValueError("transport_num_retries must be a non-negative integer")
        self.transport_num_retries = num_retries

    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        if not spec.env_key:
            # 仅 OAuth/Provider 使用的 spec，例如 openai_codex。
            return

        # 网关/本地部署覆盖现有 env；标准 Provider 不覆盖。
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # 解析 env_extras 占位符：
        #   {api_key}  → 用户 API key
        #   {api_base} → 用户 api_base，缺失时回退到 spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # 网关模式：应用网关前缀，跳过 Provider 专属前缀。
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model

        # 标准模式：为已知 Provider 自动添加前缀。
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            model = self._canonicalize_explicit_prefix(model, spec.name, spec.litellm_prefix)
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"

        return model

    @staticmethod
    def _canonicalize_explicit_prefix(model: str, spec_name: str, canonical_prefix: str) -> str:
        """Normalize explicit provider prefixes like `github-copilot/...`."""
        if "/" not in model:
            return model
        prefix, remainder = model.split("/", 1)
        if prefix.lower().replace("-", "_") != spec_name:
            return model
        return f"{canonical_prefix}/{remainder}"

    def _supports_cache_control(self, model: str) -> bool:
        """Return True when the provider supports cache_control on content blocks."""
        if self._gateway is not None:
            return self._gateway.supports_prompt_caching
        spec = find_by_model(model)
        return spec is not None and spec.supports_prompt_caching

    def supports_explicit_cache_control(self, model: str) -> bool:
        return self._supports_cache_control(model)

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Return copies of messages and tools with cache_control injected."""
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if isinstance(content, str):
                    new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                else:
                    new_content = list(content)
                    new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
                new_messages.append({**msg, "content": new_content})
            else:
                new_messages.append(msg)

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}

        return new_messages, new_tools

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return

    @staticmethod
    def _extra_msg_keys(original_model: str, resolved_model: str) -> frozenset[str]:
        """Return provider-specific extra keys to preserve in request messages."""
        spec = find_by_model(original_model) or find_by_model(resolved_model)
        if (
            (spec and spec.name == "anthropic")
            or "claude" in original_model.lower()
            or resolved_model.startswith("anthropic/")
        ):
            return _ANTHROPIC_EXTRA_KEYS
        return frozenset()

    @staticmethod
    def _requires_reasoning_content_replay(original_model: str, resolved_model: str) -> bool:
        spec = find_by_model(original_model) or find_by_model(resolved_model)
        return spec is not None and spec.requires_reasoning_content_replay

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """Normalize tool_call_id to a provider-safe 9-char alphanumeric form."""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    @staticmethod
    def _sanitize_messages(
        messages: list[dict[str, Any]],
        extra_keys: frozenset[str] = frozenset(),
        require_reasoning_content_replay: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip non-standard keys and ensure assistant messages have a content key."""
        allowed = _ALLOWED_MSG_KEYS | extra_keys
        sanitized = LLMProvider._sanitize_request_messages(messages, allowed)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return id_map.setdefault(value, LiteLLMProvider._normalize_tool_call_id(value))

        for clean in sanitized:
            if (
                require_reasoning_content_replay
                and clean.get("role") == "assistant"
                and clean.get("tool_calls")
                and clean.get("reasoning_content") is None
            ):
                clean["reasoning_content"] = ""

            # 缩短后保持 assistant tool_calls[].id 与 tool tool_call_id 同步，
            # 否则严格 Provider 会拒绝断裂的关联。
            if isinstance(clean.get("tool_calls"), list):
                normalized_tool_calls = []
                for tc in clean["tool_calls"]:
                    if not isinstance(tc, dict):
                        normalized_tool_calls.append(tc)
                        continue
                    tc_clean = dict(tc)
                    tc_clean["id"] = map_id(tc_clean.get("id"))
                    normalized_tool_calls.append(tc_clean)
                clean["tool_calls"] = normalized_tool_calls

            if clean.get("tool_call_id"):
                clean["tool_call_id"] = map_id(clean["tool_call_id"])
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)
        extra_msg_keys = self._extra_msg_keys(original_model, model)
        require_reasoning_content_replay = self._requires_reasoning_content_replay(original_model, model)

        if self._supports_cache_control(original_model) and not self.disable_auto_cache_control:
            messages, tools = self._apply_cache_control(messages, tools)

        # max_tokens 下限为 1；负数或零会让 LiteLLM 以
        # "max_tokens must be at least 1" 拒绝请求。
        max_tokens = max(1, max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(
                self._sanitize_empty_content(messages),
                extra_keys=extra_msg_keys,
                require_reasoning_content_replay=require_reasoning_content_replay,
            ),
            "max_tokens": max_tokens,
            "temperature": temperature,
            # 限制单次 LLM 调用时长，避免上游端点半关闭或从不发送 FIN 时 Agent
            # 永久悬挂。LiteLLM 默认 6000 秒，对交互式 Agent Loop 过长；600 秒
            # 足以覆盖缓慢的重推理生成，同时能让卡住的连接快速失败，供
            # chat_with_retry 重试或放弃。
            "timeout": 600,
        }

        # 应用模型专属覆盖，例如 kimi-k2.5 temperature。
        self._apply_model_overrides(model, kwargs)

        # 直接传入 api_key，比只依赖环境变量更可靠。
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # 为自定义端点传入 api_base。
        if self.api_base:
            kwargs["api_base"] = self.api_base

        # 传入额外 header，例如 AiHubMix 的 APP-Code。
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        # 传入 Provider 专属 body 扩展，例如 OpenRouter 路由固定参数。
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.transport_num_retries is not None:
            kwargs["num_retries"] = self.transport_num_retries

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # 把错误作为 content 返回以便平稳处理，但要在实时异常
            # （status_code + type）退化成字符串前完成分类；重试/fallback 层读取该结论。
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
                error_classification=self.classify_error(e),
            )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Streaming counterpart to chat().

        Yields one StreamDelta per non-empty chunk. Signature matches chat()
        so callers can swap providers transparently. The existing chat() is
        NOT modified — non-TUI paths (channels / Cron / ...)
        continue to use chat() with no behavioral change.

        Provider-specific chunk shapes (e.g. dashscope) are handled inside
        `_normalize_stream_chunk`. The default OpenAI shape extraction lives
        in that hook; subclasses or implementer additions can override.
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)
        extra_msg_keys = self._extra_msg_keys(original_model, model)
        require_reasoning_content_replay = self._requires_reasoning_content_replay(original_model, model)

        if self._supports_cache_control(original_model) and not self.disable_auto_cache_control:
            messages, tools = self._apply_cache_control(messages, tools)

        max_tokens = max(1, max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(
                self._sanitize_empty_content(messages),
                extra_keys=extra_msg_keys,
                require_reasoning_content_replay=require_reasoning_content_replay,
            ),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            # OpenAI-compatible Provider 只有在显式请求 usage 时才发出末尾 usage
            # chunk；否则 stream 不含 token 计数，下游成本/上下文追踪会看到零。
            "stream_options": {"include_usage": True},
        }

        self._apply_model_overrides(model, kwargs)

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.transport_num_retries is not None:
            kwargs["num_retries"] = self.transport_num_retries

        response = await acompletion(**kwargs)
        async for chunk in response:
            delta = self._normalize_stream_chunk(chunk)
            if delta is not None:
                yield delta

    def _normalize_stream_chunk(self, chunk: Any) -> StreamDelta | None:
        """Normalize a raw provider chunk into a StreamDelta.

        Default: OpenAI shape — `chunk.choices[0].delta.content` (str | None),
        `delta.tool_calls` (list | None), and a final `chunk.usage` snapshot
        on the trailing chunk for some providers. Returns None when the chunk
        carries no content / tool_call / usage payload so callers can skip.

        Provider-specific shapes (e.g. Qwen dashscope) are decided at
        implementation time after a real-provider smoke test (per design.md
        §D4 + tasks.md T3.4). Add a hardcoded branch here keyed on
        `self._gateway` / `find_by_model(...).name` if/when needed.
        """
        try:
            usage = getattr(chunk, "usage", None)
            usage_dict: dict[str, Any] | None = None
            if usage is not None:
                usage_dict = _normalize_usage_payload(usage)
            choices = getattr(chunk, "choices", None)
            if not choices:
                if usage_dict is None:
                    return None
                return StreamDelta(
                    content=None,
                    usage=usage_dict,
                    model=getattr(chunk, "model", None) or None,
                )
            delta_obj = getattr(choices[0], "delta", None)
            if delta_obj is None:
                return None
            content = getattr(delta_obj, "content", None)
            tool_calls = getattr(delta_obj, "tool_calls", None)
            reasoning_content = getattr(delta_obj, "reasoning_content", None) or None

            tool_call_delta: dict[str, Any] | None = None
            if tool_calls:
                # 把原始 tool_call delta 作为字典快照列表暴露，供下游重组；
                # 此处有意只做轻量处理，完整工具调用累积属于消费者职责。
                serialized = []
                for tc in tool_calls:
                    try:
                        serialized.append(tc.model_dump())  # 使用 Pydantic v2 接口
                    except AttributeError:
                        serialized.append(
                            {
                                "index": getattr(tc, "index", None),
                                "id": getattr(tc, "id", None),
                                "function": {
                                    "name": getattr(getattr(tc, "function", None), "name", None),
                                    "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                                },
                            }
                        )
                tool_call_delta = {"tool_calls": serialized}

            if content is None and tool_call_delta is None and usage_dict is None and reasoning_content is None:
                return None

            return StreamDelta(
                content=content,
                tool_call_delta=tool_call_delta,
                usage=usage_dict,
                reasoning_content=reasoning_content,
                model=getattr(chunk, "model", None) or None,
            )
        except (AttributeError, IndexError):
            return None

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        content = message.content
        finish_reason = choice.finish_reason

        # 部分 Provider（如 GitHub Copilot）会把 content 和 tool_calls 拆到多个
        # choice 中；将其合并，避免丢失 tool_calls。
        raw_tool_calls = []
        for ch in response.choices:
            msg = ch.message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                raw_tool_calls.extend(msg.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and msg.content:
                content = msg.content

        if len(response.choices) > 1:
            logger.debug(
                "LiteLLM response has {} choices, merged {} tool_calls", len(response.choices), len(raw_tool_calls)
            )

        tool_calls = []
        for tc in raw_tool_calls:
            # 必要时从 JSON 字符串解析参数。
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)

            provider_specific_fields = getattr(tc, "provider_specific_fields", None) or None
            function_provider_specific_fields = getattr(tc.function, "provider_specific_fields", None) or None

            tool_calls.append(
                ToolCallRequest(
                    id=_short_tool_id(),
                    name=tc.function.name,
                    arguments=args,
                    provider_specific_fields=provider_specific_fields,
                    function_provider_specific_fields=function_provider_specific_fields,
                )
            )

        usage = _normalize_usage_payload(response.usage) if getattr(response, "usage", None) else {}

        reasoning_content = getattr(message, "reasoning_content", None) or None
        thinking_blocks = getattr(message, "thinking_blocks", None) or None

        return LLMResponse(
            content=content,
            model=getattr(response, "model", None) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
