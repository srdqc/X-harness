"""Single source of truth for LLM call cost estimation.

Used by CallEfficiency, Tracing, and historical TokenWise compatibility code.
Returning a consistent cost from one place prevents drift between recorded and
reported estimates.

Pricing sources (in order):
    1. ``litellm.cost_per_token`` — covers most public models. Tries the
       ``openrouter/<model>`` alias first, then the bare model id.
    2. A previously refreshed OpenRouter model catalog — local memory or disk
       only on the call path. Remote refresh is an explicit operator action.
    3. ``_FALLBACK_PRICING`` — manual rate table for models missing from
       both.
    4. ``None`` — model unknown to all. Caller should degrade gracefully.

Anthropic ephemeral cache pricing is applied on top of the base rate:
    cache read  → 10% of prompt rate
    cache write → 125% of prompt rate (ephemeral 5-min TTL)

DeepSeek V4 instead uses its published automatic disk-cache hit, miss, and
output rates. Other providers pass ``cache_read_tokens=0``,
``cache_write_tokens=0`` and the function collapses to the standard formula.
"""

from __future__ import annotations

import time

import httpx
from loguru import logger

from pico.call_efficiency import model_catalog_cache

# 费率对：(prompt_cost_per_token, completion_cost_per_token)，单位为美元。
# 保持此表精简：它只为 LiteLLM 尚未收录的新模型提供回退。
# 添加条目前应先检查 LiteLLM。
_FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    # OpenRouter 模型页面快照（2026-03）。
    "z-ai/glm-4.5-air": (0.13e-6, 0.85e-6),  # 每百万令牌 $0.13/$0.85
}

# DeepSeek 2026-08-06“模型与定价”页面的快照。这些 Provider 直连费率
# 优先于 LiteLLM，因为普通的 prompt/completion 费率对无法表达缓存命中价格。
_DEEPSEEK_V4_PRICING: dict[str, tuple[float, float, float]] = {
    # 依次为未命中缓存输入、命中缓存输入、输出，单位为美元/令牌。
    "deepseek-v4-flash": (0.14e-6, 0.0028e-6, 0.28e-6),
    "deepseek-v4-pro": (0.435e-6, 0.003625e-6, 0.87e-6),
}

# 记录已警告过的未知模型，确保每个模型只输出一次日志。
_WARNED_UNKNOWN: set[str] = set()

# 实时 OpenRouter 价格表，惰性获取并在进程内缓存 1 小时。
# 同时映射完整 id（``deepseek/deepseek-v4-pro``）和不带命名空间的别名。
# （``deepseek-v4-pro``）映射到 OpenRouter 的逐令牌 ``pricing`` 字典。
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_CACHE_TTL = 3600
_OPENROUTER_CACHE: dict[str, dict] = {}
_OPENROUTER_CACHE_TIME: float = 0.0
_ALLOW_NETWORK_CATALOG = False


def _try_litellm_rates(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    allow_import: bool,
) -> tuple[float, float] | None:
    """Ask LiteLLM for per-token rates. Returns (prompt_rate, completion_rate) or None."""
    import sys

    if not allow_import and "litellm" not in sys.modules:
        return None
    try:
        from pico.providers.litellm_setup import import_litellm

        litellm = import_litellm()
    except Exception:
        return None

    candidates = [model]
    if not model.startswith("openrouter/"):
        candidates.insert(0, f"openrouter/{model}")

    # litellm.cost_per_token 至少需要 1 个非零 token 才能计算。
    # 这里传入合成 token 来反推出单 token 费率。
    probe_in = input_tokens if input_tokens else 1
    probe_out = output_tokens if output_tokens else 1

    for candidate in candidates:
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=candidate, prompt_tokens=probe_in, completion_tokens=probe_out
            )
        except Exception:
            continue
        if prompt_cost is None or completion_cost is None:
            continue
        if prompt_cost == 0 and completion_cost == 0:
            # LiteLLM 对未知模型返回 (0, 0)，应按未命中处理。
            continue
        return prompt_cost / probe_in, completion_cost / probe_out

    return None


def _fetch_openrouter_models() -> dict[str, dict]:
    """Return OpenRouter's model table, fetched live and cached 1h in-process.

    Each entry is ``{"pricing": ..., "context_length": ...}``, double-keyed by
    full id and bare alias. On any network failure, returns the stale cache
    (or an empty dict) — pricing must never raise into the cost path.
    """
    global _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME

    now = time.time()
    if _OPENROUTER_CACHE and (now - _OPENROUTER_CACHE_TIME) < _OPENROUTER_CACHE_TTL:
        return _OPENROUTER_CACHE

    # 磁盘层：从仍然新鲜的磁盘缓存热启动（也可复用同级进程刚获取的数据），
    # 无需访问网络。
    disk = model_catalog_cache.load()
    if disk is not None and (now - disk[1]) < _OPENROUTER_CACHE_TTL:
        _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME = disk
        return _OPENROUTER_CACHE

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(_OPENROUTER_MODELS_URL)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("pricing: OpenRouter models fetch failed ({}), degrading", exc)
        if _OPENROUTER_CACHE:
            return _OPENROUTER_CACHE
        if disk is not None:
            _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME = disk
            return _OPENROUTER_CACHE
        return {}

    cache: dict[str, dict] = {}
    for model in data.get("data", []):
        model_id = model.get("id", "")
        if not model_id:
            continue
        entry = {
            "pricing": model.get("pricing") or {},
            "context_length": model.get("context_length"),
        }
        cache[model_id] = entry
        if "/" in model_id:
            cache.setdefault(model_id.split("/", 1)[1], entry)

    _OPENROUTER_CACHE = cache
    _OPENROUTER_CACHE_TIME = time.time()
    model_catalog_cache.save(cache)
    return cache


def _cached_openrouter_models() -> dict[str, dict]:
    global _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME

    if _OPENROUTER_CACHE:
        return _OPENROUTER_CACHE
    disk = model_catalog_cache.load()
    if disk is not None:
        _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME = disk
    return _OPENROUTER_CACHE


def _lookup_openrouter_entry(model: str, *, allow_network: bool) -> dict | None:
    """Resolve a model to its OpenRouter catalog entry.

    Strips a leading ``openrouter/`` then tries the remaining id and its bare
    alias. Used as a cross-provider fallback for any model LiteLLM doesn't map,
    so the catalog also covers e.g. a direct ``deepseek/...`` route.
    """
    key = model.removeprefix("openrouter/")
    table = _fetch_openrouter_models() if allow_network else _cached_openrouter_models()
    entry = table.get(key)
    if entry is None and "/" in key:
        entry = table.get(key.split("/", 1)[1])
    return entry


def _try_openrouter_rates(model: str, *, allow_network: bool) -> tuple[float, float] | None:
    """Look up optional OpenRouter catalog rates. Returns rates or None."""
    entry = _lookup_openrouter_entry(model, allow_network=allow_network)
    if not entry:
        return None
    pricing = entry.get("pricing") or {}
    try:
        return float(pricing["prompt"]), float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None


def _try_litellm_context_window(model: str, *, allow_import: bool) -> int | None:
    """LiteLLM's static model metadata — offline, covers most mapped providers."""
    import sys

    if not allow_import and "litellm" not in sys.modules:
        return None
    try:
        from pico.providers.litellm_setup import import_litellm

        litellm = import_litellm()
    except Exception:
        return None

    candidates = [model]
    if not model.startswith("openrouter/"):
        candidates.insert(0, f"openrouter/{model}")

    for candidate in candidates:
        try:
            info = litellm.get_model_info(candidate)
        except Exception:
            continue
        if not info:
            continue
        window = info.get("max_input_tokens") or info.get("max_tokens")
        if window:
            try:
                return int(window)
            except (TypeError, ValueError):
                continue
    return None


def resolve_context_window(
    model: str,
    *,
    allow_network: bool | None = None,
    allow_litellm_import: bool = True,
) -> int | None:
    """Return a model's real context window in tokens, or None.

    Sources, in order: LiteLLM's static model metadata, then a locally cached
    OpenRouter catalog. The default call path never refreshes that catalog over
    the network. Unknown models return None so the caller keeps its configured
    default.
    """
    window = _try_litellm_context_window(model, allow_import=allow_litellm_import)
    if window:
        return window

    network = _ALLOW_NETWORK_CATALOG if allow_network is None else allow_network
    entry = _lookup_openrouter_entry(model, allow_network=network)
    if entry:
        try:
            length = int(entry.get("context_length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length:
            return length
    return None


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    *,
    allow_network: bool | None = None,
    allow_litellm_import: bool = True,
) -> float | None:
    """Estimate USD cost for a single LLM call. Returns None for unknown models.

    ``input_tokens`` is fresh (non-cache) prompt tokens. Anthropic's
    ``usage.input_tokens`` already excludes cache tokens, so pass it through
    untouched. DeepSeek reports cache hits separately from total prompt tokens;
    callers normalize the total to fresh tokens before invoking this function.
    """
    deepseek_model = model.removeprefix("deepseek/")
    if not model.startswith("openrouter/") and deepseek_model in _DEEPSEEK_V4_PRICING:
        miss_rate, hit_rate, output_rate = _DEEPSEEK_V4_PRICING[deepseek_model]
        return (
            input_tokens * miss_rate
            + output_tokens * output_rate
            + cache_read_tokens * hit_rate
            + cache_write_tokens * miss_rate
        )

    network = _ALLOW_NETWORK_CATALOG if allow_network is None else allow_network
    rates = _try_litellm_rates(
        model,
        input_tokens,
        output_tokens,
        allow_import=allow_litellm_import,
    )
    if rates is None:
        rates = _try_openrouter_rates(model, allow_network=network)
    if rates is None:
        key = model.removeprefix("openrouter/")
        if key in _FALLBACK_PRICING:
            rates = _FALLBACK_PRICING[key]
        else:
            if model not in _WARNED_UNKNOWN:
                logger.warning("pricing: unknown model '{}', cost estimate = None", model)
                _WARNED_UNKNOWN.add(model)
            return None

    prompt_rate, completion_rate = rates
    cost = (
        input_tokens * prompt_rate
        + output_tokens * completion_rate
        + cache_read_tokens * prompt_rate * 0.1
        + cache_write_tokens * prompt_rate * 1.25
    )
    return cost


def estimate_cost_from_rates(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    input_usd_per_token: float,
    output_usd_per_token: float,
    cache_read_usd_per_token: float,
    cache_write_usd_per_token: float | None = None,
) -> float:
    """Estimate one call from a frozen evidence price snapshot."""
    write_rate = input_usd_per_token if cache_write_usd_per_token is None else cache_write_usd_per_token
    return (
        input_tokens * input_usd_per_token
        + output_tokens * output_usd_per_token
        + cache_read_tokens * cache_read_usd_per_token
        + cache_write_tokens * write_rate
    )


def reset_warning_cache() -> None:
    """Clear the set of models we've already logged an 'unknown' warning for.

    Only useful for tests — production code should let warnings land once.
    """
    _WARNED_UNKNOWN.clear()


def reset_openrouter_cache() -> None:
    """Clear the in-process OpenRouter catalog cache.

    Only useful for tests — pair it with the ``model_catalog_cache._CACHE_PATH``
    seam to exercise the disk tiers without touching the real ~/.pico/cache/.
    """
    global _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME
    _OPENROUTER_CACHE = {}
    _OPENROUTER_CACHE_TIME = 0.0


def refresh_model_catalog() -> dict[str, dict]:
    """Explicitly refresh the optional OpenRouter model catalog over the network."""
    return _fetch_openrouter_models()
