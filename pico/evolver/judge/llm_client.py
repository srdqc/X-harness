"""LLM client for the judge — pluggable backends + mix routing.

The judge has two distinct workloads:

- **L1 detection** (is this trajectory hosed by an infrastructure bug?)
  — pattern recognition, cheap model suffices.
- **L2 / L3 patch proposal** (which file to edit, what change) — needs
  stronger reasoning; pay for a larger model.

We support three operating modes, controllable via :class:`JudgeLLMConfig`:

- ``"single"`` — one backend handles everything. Pick a model strong
  enough for patches; it will also do L1 detection. Useful for ablations.
- ``"two_step"`` *(default mix mode)* — cheap L1 backend runs first; if
  it determines L1, we return its output and skip the expensive call.
  Otherwise we discard its patch guess and call the strong patch backend.
- ``"pure_<name>"`` — alias for ``single`` with both slots configured to
  the same backend (just a convenience for "use only Qwen" or "use only
  OpenRouter").

Backend implementations:

- :class:`LitellmBackend` wraps any ``LLMProvider`` from
  ``pico.providers`` — works with self-hosted Qwen-397B, plus any
  other model the existing provider stack supports.
- :class:`OpenRouterBackend` issues direct HTTPS to
  ``openrouter.ai/api/v1`` — gives access to Claude / GPT / Gemini /
  Qwen / DeepSeek through one API.
- :class:`MockBackend` returns canned responses; used by every test in
  this module and any downstream test that needs a deterministic judge.

Configuration is data-driven (:func:`load_judge_config`) so the same
code path runs in production (real backends), in tests (MockBackend),
and in ablations (config-flip changes mode).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .parser import JudgeParseError, parse_judge_output
from .prompts import build_judge_messages
from .schema import IssueType, JudgeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 后端协议与实现
# ---------------------------------------------------------------------------


class JudgeLLMBackend(ABC):
    """Minimum surface a backend must expose.

    The judge does not need streaming, tool calls, or function-calling —
    just plain chat completion that returns the assistant's text. Keep
    the interface narrow so adding a new backend (Vertex / Together /
    a local vLLM endpoint) requires <30 lines.
    """

    name: str  # 由子类设置

    @abstractmethod
    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        """Run one chat completion. Returns the assistant's text body."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class LitellmBackend(JudgeLLMBackend):
    """Backend that delegates to a Pico ``LLMProvider``.

    Used for the self-hosted Qwen-397B path (which Pico's LiteLLM
    provider already routes). The provider is lazy-imported so this
    module doesn't drag the full provider stack into pure-unit-test
    environments.

    The wrapped provider must expose ``async def chat(messages, ...)``
    returning an object with a ``.content`` string attribute (the
    standard ``LLMResponse``). We strip everything else.
    """

    def __init__(
        self,
        provider: Any,  # pico.providers.base.LLMProvider，延迟确定类型
        *,
        model: Optional[str] = None,
        name: str = "litellm",
    ) -> None:
        self._provider = provider
        self._model = model
        self.name = name

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        # 经由 base.LLMProvider 的 chat_with_retry 路由，使评判器继承智能体循环已依赖的空响应
        # 重试和同步 OpenAI 回退。直接调用 provider.chat() 会在评判层暴露提供商空内容错误，
        # 早期干跑约占 3%。提供商的 chat[_with_retry] 接受 model=None 并回退到配置默认值；
        # 只有显式指定时才传入覆盖值。
        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if self._model:
            kwargs["model"] = self._model
        response = await self._provider.chat_with_retry(messages, **kwargs)
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise RuntimeError(
                f"LitellmBackend({self.name}): provider returned non-string content"
                f" ({type(content).__name__}); cannot parse as judge output."
            )
        return content


class OpenRouterBackend(JudgeLLMBackend):
    """Direct HTTP backend for OpenRouter (``openrouter.ai``).

    Why direct HTTP instead of routing through litellm? OpenRouter API
    keys live in a separate env var from pico's main provider stack,
    and we want the judge's external-LLM budget to be visible and
    accounted independently — easier to enforce a hard cap and to swap
    models per ablation. The HTTP shape is OpenAI-compatible, so the
    code is small.

    Model strings use OpenRouter's namespaced form, e.g.:

    - ``"anthropic/claude-haiku-4-5"``
    - ``"openai/gpt-4.1-mini"``
    - ``"google/gemini-2.5-flash"``
    - ``"qwen/qwen3-235b"``

    API key resolution: explicit ``api_key`` arg → env var ``api_key_env``
    → ``OPENROUTER_API_KEY``. Raises at call time if none found.

    ``httpx`` is required at call time but imported lazily so unit tests
    that only use ``MockBackend`` don't need it installed.
    """

    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
    DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
    # 对空内容和暂时性 HTTP 错误重试。OpenRouterBackend 绕过 Pico 提供商栈，无法直接获得
    # chat_with_retry，因此添加最小内部重试以保持一致。
    _RETRY_DELAYS = (1.0, 2.0, 4.0)

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout_seconds: float = 60.0,
        name: str = "openrouter",
    ) -> None:
        self._model = model
        self._explicit_key = api_key
        self._api_key_env = api_key_env or self.DEFAULT_API_KEY_ENV
        self._api_base = (api_base or self.DEFAULT_API_BASE).rstrip("/")
        self._timeout = timeout_seconds
        self.name = name

    def _resolve_api_key(self) -> str:
        if self._explicit_key:
            return self._explicit_key
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"OpenRouterBackend({self.name}): no API key — pass api_key, or set env {self._api_key_env}"
            )
        return key

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        # 延迟导入：不访问 OpenRouter 的测试无需 httpx。
        import asyncio  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        api_key = self._resolve_api_key()
        url = f"{self._api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter 建议标识调用方，以便进行路由分析：
            "HTTP-Referer": "https://gitee.com/htxoffical/pico-harness",
            "X-Title": "Pico Harness Evolver",
        }

        # 对空内容和暂时性 HTTP 错误重试。LitellmBackend 通过 provider.chat_with_retry 自动
        # 获得重试；OpenRouter 绕过提供商栈，因此在此添加同样“检测空值 -> 退避 -> 重试”形态的
        # 最小原地重试。总尝试次数为 len(_RETRY_DELAYS) 加最后一次，共 4 次。
        last_exc: Exception | None = None
        last_data: dict[str, Any] | None = None
        for attempt, delay in enumerate(self._RETRY_DELAYS, start=1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                content = self._extract_content(data)
                if content and content.strip():
                    return content
                # 空内容——退避后重试。
                last_data = data
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                last_exc = exc
            await asyncio.sleep(delay)

            # 最后一次尝试不再抑制异常。
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        content = self._extract_content(data)
        if content and content.strip():
            return content
        # 全部重试后仍为空，抛出异常让批处理记录错误。
        if last_exc is not None:
            raise RuntimeError(
                f"OpenRouterBackend({self.name}): empty content after "
                f"{len(self._RETRY_DELAYS) + 1} attempts; last exc: {last_exc!r}"
            ) from last_exc
        raise RuntimeError(
            f"OpenRouterBackend({self.name}): empty content after "
            f"{len(self._RETRY_DELAYS) + 1} attempts; last payload: "
            f"{(last_data or data)!r}"
        )

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> Optional[str]:
        """Pull message content from an OpenAI-shaped response, or None."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None


class MockBackend(JudgeLLMBackend):
    """In-memory backend that returns scripted responses, for tests.

    Pass a list of strings in ``responses``; each call pops the next one
    in order. Raises ``IndexError`` if the test under-scripts responses
    — fail loud rather than silently re-using the last.

    ``calls`` records every (messages, max_tokens, temperature) tuple for
    assertions: tests can verify the backend was invoked the right number
    of times with the expected payload.
    """

    def __init__(self, responses: list[str], *, name: str = "mock") -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.name = name

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self._responses:
            raise IndexError(
                f"MockBackend({self.name}): no more scripted responses; received {len(self.calls)} call(s) total"
            )
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# 配置数据类与门面
# ---------------------------------------------------------------------------


Mode = Literal["single", "two_step"]
TrajectoryFormat = Literal["full", "compressed"]


@dataclass
class JudgeLLMConfig:
    """Configuration for :class:`JudgeLLM`.

    The two backend slots are populated externally — we don't construct
    backends from this config, so the config stays serialisation-friendly
    (yaml/json) and the test harness can inject a ``MockBackend`` without
    going through any factory.

    Trajectory format (full vs compressed) is configured per backend slot:

    - ``l1_trajectory_format``: what the L1-detection backend sees. Default
      ``"compressed"`` — L1 signals (empty content / docker errors /
      pattern repetition) are visible in a ~10K compression; paying for
      150K full context here is waste.
    - ``patch_trajectory_format``: what the patch-proposal backend sees.
      Default ``"full"`` — writing a good patch needs concrete tool calls
      and detailed reasoning steps that compression can blur.

    Caller supplies both ``trajectory_text`` (full) and optional
    ``trajectory_text_compressed`` to :meth:`JudgeLLM.judge`. When a
    backend slot's format is ``"compressed"`` but no compressed text was
    passed, we fall back to full with a debug log (so misconfiguration
    is visible but doesn't crash).

    Use :func:`build_judge_llm` to assemble a ``JudgeLLM`` from a
    config-dict that *does* describe backends. That function is the
    integration point with yaml/json config files.
    """

    mode: Mode = "two_step"
    l1_trajectory_format: TrajectoryFormat = "compressed"
    patch_trajectory_format: TrajectoryFormat = "full"
    max_tokens: int = 4000
    temperature: float = 0.0
    # 诊断：记录每次调用的输入长度和所用后端。默认关闭，使测试标准输出保持安静。
    debug_log: bool = False


class JudgeLLM:
    """Facade that orchestrates one judge analysis.

    Single-call mode is straightforward: build the (system, user)
    messages, send to ``patch_backend``, parse and return.

    Two-step mode is the default mix: ``l1_backend`` makes the first
    pass. If it judges L1 (infrastructure bug), we return its result
    immediately — saves the expensive patch_backend call. If it judges
    L2/L3, we discard its patch guess (cheap models tend to be sloppy
    on patch proposals) and call ``patch_backend`` for a higher-quality
    rewrite.

    Why discard the cheap model's patch instead of keeping it when good?
    Reliability of the patch is downstream-critical (a bad patch wastes
    evaluation budget). The cheap model's L1 detection is high-recall;
    its patch field is low-precision. Splitting trust along this axis
    is what makes the mix worth its complexity.
    """

    def __init__(
        self,
        l1_backend: JudgeLLMBackend,
        patch_backend: JudgeLLMBackend,
        config: Optional[JudgeLLMConfig] = None,
    ) -> None:
        self._l1 = l1_backend
        self._patch = patch_backend
        self._config = config or JudgeLLMConfig()

    @property
    def config(self) -> JudgeLLMConfig:
        return self._config

    async def judge(
        self,
        trajectory_id: str,
        task_description: str,
        trajectory_text: str,
        trajectory_text_compressed: Optional[str] = None,
    ) -> JudgeResult:
        """Run the full judge pipeline on one trajectory.

        ``trajectory_text`` is the full original trajectory (required).
        ``trajectory_text_compressed`` is an optional pre-compressed
        summary (~10K tokens, "agent debugger" style). When provided AND
        a backend slot's ``*_trajectory_format`` is ``"compressed"``, that
        backend receives the compressed text; otherwise it falls back to
        full (with a debug log when ``debug_log=True``).
        """
        l1_text = self._select_trajectory_text(
            self._config.l1_trajectory_format,
            trajectory_text,
            trajectory_text_compressed,
            slot="l1",
        )
        patch_text = self._select_trajectory_text(
            self._config.patch_trajectory_format,
            trajectory_text,
            trajectory_text_compressed,
            slot="patch",
        )

        patch_messages = build_judge_messages(
            trajectory_id=trajectory_id,
            task_description=task_description,
            trajectory_text=patch_text,
        )
        if self._config.mode == "single":
            return await self._call_and_parse(self._patch, patch_messages, trajectory_id)

        # 两步模式
        l1_messages = build_judge_messages(
            trajectory_id=trajectory_id,
            task_description=task_description,
            trajectory_text=l1_text,
        )
        l1_result = await self._call_and_parse(self._l1, l1_messages, trajectory_id)
        if l1_result.issue_type == IssueType.L1:
            if self._config.debug_log:
                logger.info(
                    "judge: %s → L1 detected by l1_backend, skipping patch_backend",
                    trajectory_id,
                )
            return l1_result
        if self._config.debug_log:
            logger.info(
                "judge: %s → %s detected, calling patch_backend",
                trajectory_id,
                l1_result.issue_type.value,
            )
        return await self._call_and_parse(self._patch, patch_messages, trajectory_id)

    def _select_trajectory_text(
        self,
        want: TrajectoryFormat,
        full: str,
        compressed: Optional[str],
        *,
        slot: str,
    ) -> str:
        """Pick which trajectory text to feed a backend, with fallback.

        ``want`` is what the config requests; if it's ``"compressed"`` but
        no compressed text was provided, we fall back to ``full`` so the
        judge still runs (cost balloons but correctness is preserved).
        Logged at debug level so misconfig is visible.
        """
        if want == "compressed":
            if compressed is None:
                if self._config.debug_log:
                    logger.warning(
                        "judge[%s]: config wants compressed trajectory but none "
                        "provided; falling back to full (cost may be higher than "
                        "expected)",
                        slot,
                    )
                return full
            return compressed
        return full

    async def _call_and_parse(
        self,
        backend: JudgeLLMBackend,
        messages: list[dict[str, str]],
        trajectory_id: str,
    ) -> JudgeResult:
        raw = await backend.call(
            messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        try:
            return parse_judge_output(raw, expected_trajectory_id=trajectory_id)
        except JudgeParseError:
            logger.warning(
                "judge.parse failed on backend=%s, traj=%s; raw=%s",
                backend.name,
                trajectory_id,
                raw[:500],
            )
            raise

    def __repr__(self) -> str:
        return f"JudgeLLM(mode={self._config.mode!r}, l1={self._l1!r}, patch={self._patch!r})"


# ---------------------------------------------------------------------------
# 配置驱动构建器（yaml/json 的集成点）
# ---------------------------------------------------------------------------


def build_backend(spec: dict[str, Any]) -> JudgeLLMBackend:
    """Build one backend from a config-dict.

    Supported ``type`` values:

    - ``"openrouter"`` — :class:`OpenRouterBackend`. Required: ``model``.
      Optional: ``api_key``, ``api_key_env``, ``api_base``,
      ``timeout_seconds``, ``name``.
    - ``"litellm"`` — :class:`LitellmBackend`. Required: ``provider``
      (an already-instantiated ``LLMProvider``). Optional: ``model``,
      ``name``. **Cannot be built from pure dict** because the provider
      object must be constructed elsewhere — callers pass it in via
      ``spec["provider"]`` (typically the same provider AgentLoop uses).
    - ``"mock"`` — :class:`MockBackend`. Required: ``responses`` (list
      of strings). For tests only.

    Raises ``ValueError`` on unknown ``type``.
    """
    backend_type = spec.get("type")
    if backend_type == "openrouter":
        return OpenRouterBackend(
            model=spec["model"],
            api_key=spec.get("api_key"),
            api_key_env=spec.get("api_key_env"),
            api_base=spec.get("api_base"),
            timeout_seconds=spec.get("timeout_seconds", 60.0),
            name=spec.get("name", "openrouter"),
        )
    if backend_type == "litellm":
        if "provider" not in spec:
            raise ValueError(
                "litellm backend spec requires a pre-built 'provider' object; cannot construct from pure dict"
            )
        return LitellmBackend(
            provider=spec["provider"],
            model=spec.get("model"),
            name=spec.get("name", "litellm"),
        )
    if backend_type == "mock":
        return MockBackend(
            responses=list(spec.get("responses", [])),
            name=spec.get("name", "mock"),
        )
    raise ValueError(f"unknown backend type {backend_type!r}; supported: openrouter, litellm, mock")


def build_judge_llm(spec: dict[str, Any]) -> JudgeLLM:
    """Assemble a :class:`JudgeLLM` from one dict.

    Expected shape::

        {
          "mode": "two_step",          # or "single"
          "max_tokens": 4000,
          "temperature": 0.0,
          "debug_log": false,
          "l1_backend":    { ... build_backend spec ... },
          "patch_backend": { ... build_backend spec ... },
        }

    For ``mode="single"``, ``l1_backend`` may be omitted (the
    ``patch_backend`` handles everything). If both are present in
    single mode, ``l1_backend`` is ignored.

    Convenience: setting ``l1_backend`` and ``patch_backend`` to the
    same spec is the "pure Qwen" / "pure OpenRouter" pattern — works
    without a special mode.
    """
    mode = spec.get("mode", "two_step")
    config = JudgeLLMConfig(
        mode=mode,
        l1_trajectory_format=spec.get("l1_trajectory_format", "compressed"),
        patch_trajectory_format=spec.get("patch_trajectory_format", "full"),
        max_tokens=spec.get("max_tokens", 4000),
        temperature=spec.get("temperature", 0.0),
        debug_log=spec.get("debug_log", False),
    )

    patch_spec = spec.get("patch_backend")
    if patch_spec is None:
        raise ValueError("build_judge_llm: 'patch_backend' is required")
    patch_backend = build_backend(patch_spec)

    if mode == "two_step":
        l1_spec = spec.get("l1_backend")
        if l1_spec is None:
            raise ValueError("build_judge_llm: mode='two_step' requires 'l1_backend'")
        l1_backend = build_backend(l1_spec)
    else:
        # 单后端模式：在 l1 槽复用 patch_backend，实际不会调用。
        l1_backend = patch_backend

    return JudgeLLM(
        l1_backend=l1_backend,
        patch_backend=patch_backend,
        config=config,
    )


__all__ = [
    "JudgeLLMBackend",
    "LitellmBackend",
    "OpenRouterBackend",
    "MockBackend",
    "JudgeLLMConfig",
    "JudgeLLM",
    "Mode",
    "TrajectoryFormat",
    "build_backend",
    "build_judge_llm",
]
