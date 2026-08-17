"""Role call_fn factory: yaml ``models:`` section -> {driver, design, verdict}.

Provider specs:

- ``{provider: openai_compat, base_url, model, ...}`` -> one OpenAI-compatible
  chat endpoint (:func:`pico.evolver.orchestrator.providers.openai_compat.make_call_fn`).
- ``{provider: claude_cli, model, ...}`` -> ``claude -p`` subprocess per call.
- ``{provider: pico, model?, api_base?, api_key_env?}`` -> Pico's
  ``LitellmProvider`` bridged to a sync call_fn; ``model`` omitted falls back
  to Pico's ``agents.defaults.model`` - so a config file with no
  ``models:`` section evolves with whatever model Pico itself is running.

Role fallbacks: ``design`` omitted -> reuse driver; ``verdict`` omitted ->
None (the orchestrator then drafts verdicts with the driver). Note the driver
model and the *subject's* model are different knobs: the subject agent's model
lives in the bench config and is pinned for the whole run (same-regime rule).
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

CallFn = Callable[[list], str]

_DEFAULT_SPEC = {"provider": "pico"}


def _pico_default_model() -> str:
    from pico.config.loader import load_config

    return load_config().agents.defaults.model


def make_pico_call_fn(
    model: Optional[str] = None,
    *,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> CallFn:
    from pico.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=model or _pico_default_model(),
    )

    def call(messages: list) -> str:
        # 同步桥：每次调用拥有私有事件循环，因此可从循环的工作线程安全调用，用于并行分类归纳。
        resp = asyncio.run(
            provider.chat_with_retry(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        content = getattr(resp, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Pico provider returned empty content")
        return content

    return call


def build_call_fn(spec: dict, *, role: str = "?") -> CallFn:
    try:
        return _build_call_fn(spec)
    except TypeError as exc:
        # 提供商工厂中未知或缺失的关键字参数属于配置拼写错误，而非编程错误；应以可读形式展示。
        raise ValueError(f"models.{role}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"models.{role}: {exc}") from exc


def _build_call_fn(spec: dict) -> CallFn:
    if not isinstance(spec, dict):
        raise ValueError(f"model spec must be a mapping, got {type(spec).__name__}")
    kind = spec.get("provider", "pico")
    kwargs = {k: v for k, v in spec.items() if k != "provider"}
    if kind == "openai_compat":
        from pico.evolver.orchestrator.providers.openai_compat import make_call_fn

        if "retry_delays" in kwargs:
            delays = kwargs["retry_delays"]
            if not isinstance(delays, (list, tuple)):
                raise ValueError("retry_delays must be a list of seconds")
            kwargs["retry_delays"] = tuple(delays)
        return make_call_fn(**kwargs)
    if kind == "claude_cli":
        import shutil

        from pico.evolver.orchestrator.providers.claude_cli import make_claude_call_fn

        model = kwargs.pop("model", None)
        if not model:
            raise ValueError("claude_cli spec requires 'model'")
        claude_bin = kwargs.get("claude_bin", "claude")
        if shutil.which(claude_bin) is None:
            raise ValueError(
                f"claude_cli: {claude_bin!r} not found on PATH — install the "
                "Claude Code CLI and log in, or switch this role to "
                "openai_compat/pico"
            )
        return make_claude_call_fn(model, **kwargs)
    if kind == "pico":
        model = kwargs.pop("model", None)
        return make_pico_call_fn(model, **kwargs)
    raise ValueError(f"unknown model provider {kind!r} (expected openai_compat / claude_cli / pico)")


def build_role_call_fns(models_cfg: dict) -> dict[str, Optional[CallFn]]:
    driver = build_call_fn(models_cfg.get("driver", _DEFAULT_SPEC), role="driver")
    design = build_call_fn(models_cfg["design"], role="design") if models_cfg.get("design") else driver
    verdict = build_call_fn(models_cfg["verdict"], role="verdict") if models_cfg.get("verdict") else None
    return {"driver": driver, "design": design, "verdict": verdict}


def describe_models(models_cfg: dict) -> dict:
    """Resolved model description for the run_meta snapshot (no secrets)."""
    out = {}
    for role in ("driver", "design", "verdict"):
        spec = models_cfg.get(role)
        if spec is None:
            if role == "driver":
                spec = _DEFAULT_SPEC
            elif role == "design":
                spec = {"inherit": "driver"}
            else:
                spec = {"omitted": "driver drafts verdicts"}
        out[role] = {k: v for k, v in spec.items() if "key" not in k.lower()}
        if (
            out[role].get("provider", "pico") == "pico"
            and "model" not in out[role]
            and "inherit" not in out[role]
            and "omitted" not in out[role]
        ):
            try:
                out[role]["model"] = _pico_default_model()
            except Exception:  # noqa: BLE001 — 描述仅尽力生成
                out[role]["model"] = "<pico default>"
    return out


__all__ = ["CallFn", "build_call_fn", "build_role_call_fns", "describe_models", "make_pico_call_fn"]
