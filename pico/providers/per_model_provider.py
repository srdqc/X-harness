"""Provider that dispatches each call to a per-model endpoint by model name.

Used by the ``knn`` routing backend, where routable models live on different
OpenAI-compatible endpoints. Models listed in the routing config route to their
configured endpoints; any other model name (e.g. the agent default used by
background subsystems) is served by ``fallback``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from pico.providers.base import LLMProvider, LLMResponse, StreamDelta
from pico.providers.custom_provider import CustomProvider

if TYPE_CHECKING:
    from pico.config.schema import ModelEndpoint


class PerModelProvider(LLMProvider):
    """Route provider calls to a per-model :class:`CustomProvider` by model name."""

    def __init__(self, models: "Sequence[ModelEndpoint]", fallback: LLMProvider):
        super().__init__()
        self._fallback = fallback
        self._by_model: dict[str, CustomProvider] = {
            m.model: CustomProvider(api_key=m.api_key, api_base=m.api_base, default_model=m.model)
            for m in models
            if m.model
        }
        self._default = next(iter(self._by_model), None) or fallback.get_default_model()

    def _pick(self, model: str | None) -> LLMProvider:
        return self._by_model.get(model or "", self._fallback)

    def get_default_model(self) -> str:
        return self._default

    def supports_explicit_cache_control(self, model: str) -> bool:
        return self._pick(model).supports_explicit_cache_control(model)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self._pick(model).chat(messages, tools, model=model, **kwargs)

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if not fallback_models:
            return await self._pick(model).chat_with_retry(messages, tools, model=model, **kwargs)

        model_chain = [model, *fallback_models]
        response: LLMResponse | None = None
        for index, current_model in enumerate(model_chain):
            endpoint = self._pick(current_model)
            response = await endpoint.chat_with_retry(
                messages,
                tools,
                model=current_model,
                fallback_models=None,
                **kwargs,
            )
            if response.finish_reason != "error":
                return response
            classification = response.error_classification or endpoint.classify_error(content=response.content)
            response.error_classification = classification
            if index + 1 >= len(model_chain) or not classification.should_fallback:
                return response
            logger.warning(
                "LLM call failed on model={} [{}], falling back to {}: {}",
                current_model,
                classification.category,
                model_chain[index + 1],
                (response.content or "")[:120],
            )

        return response  # type: ignore[return-value]

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        async for delta in self._pick(model).chat_stream(messages, tools, model=model, **kwargs):
            yield delta
