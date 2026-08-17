"""Empty-response recovery for the agent loop.

Pure decision logic, no I/O — the loop owns the side effects (appending
messages, incrementing counters, ``continue``). Kept separate from the loop so
the branching can be unit-tested in isolation.

A turn that ends with no visible text would otherwise surface a canned
"no response to give" dud — a zero-score turn on weaker models. This recovers
the turn before giving up, in three bounded modes:

  PREFILL  thinking-only — the model emitted only reasoning (a structured field
           or an inline <think> block) and no body. Re-feed its own reasoning so
           it continues into the answer.
  NUDGE    post-tool empty — the model ran a tool then returned nothing. Inject a
           short user nudge so it processes the tool result.
  RETRY    plain empty — re-request as-is.

This recovers an empty turn before it is ever sent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from pico.providers.base import LLMResponse

# 内容中的思考标记。部分模型网关将推理以 <think>…</think> 形式放入 ``content``，
# 而不是结构化 reasoning_content 字段，因此必须扫描内容；只检查结构化字段会漏掉它们。
_THINK_TAG_RE = re.compile(r"<think>|<thinking>|<reasoning>", re.IGNORECASE)

POST_TOOL_NUDGE = (
    "You executed tool calls but returned an empty response. Use the tool "
    "results above to continue the task, or give your final answer now."
)


class RecoveryAction(Enum):
    """What the loop should do about an empty assistant response."""

    COMPLETE = auto()  # 存在可见文本或预算耗尽 → 结束 Turn
    PREFILL = auto()  # 只有思考内容 → 回填推理后重试
    NUDGE = auto()  # 工具后空响应 → 注入空响应和用户提示后重试
    RETRY = auto()  # 普通空响应 → 原样重试


@dataclass(frozen=True)
class RecoveryLimits:
    """Per-turn retry budgets."""

    enabled: bool = True
    post_tool_empty_max_nudges: int = 1
    thinking_prefill_max_retries: int = 2
    empty_content_max_retries: int = 3


def limits_from_defaults(defaults: object) -> RecoveryLimits:
    """Build limits from an ``agents.defaults`` config object (duck-typed).

    Centralizes the config→RecoveryLimits mapping so the several AgentLoop
    construction sites don't each repeat the field plumbing.
    """
    return RecoveryLimits(
        enabled=getattr(defaults, "empty_recovery_enabled", True),
        post_tool_empty_max_nudges=getattr(defaults, "post_tool_empty_max_nudges", 1),
        thinking_prefill_max_retries=getattr(defaults, "thinking_prefill_max_retries", 2),
        empty_content_max_retries=getattr(defaults, "empty_content_max_retries", 3),
    )


def has_inline_thinking(content: str | None) -> bool:
    """True when raw content carries a <think>/<thinking>/<reasoning> marker."""
    return bool(content) and bool(_THINK_TAG_RE.search(content))


def has_thinking(response: LLMResponse) -> bool:
    """True when the response produced reasoning in any form (structured or inline)."""
    return bool(response.reasoning_content or response.thinking_blocks or has_inline_thinking(response.content))


def classify_empty_response(
    response: LLMResponse,
    visible: str | None,
    *,
    prev_had_tool_calls: bool,
    nudges_done: int,
    prefill_retries: int,
    empty_retries: int,
    limits: RecoveryLimits,
) -> RecoveryAction:
    """Decide how to handle a no-tool-call assistant response.

    ``visible`` is ``response.content`` after stripping <think> blocks — i.e. the
    user-facing text. Non-empty ``visible`` (or recovery disabled) means the turn
    is done.

    Ordering puts PREFILL before NUDGE so a thinking-only response is continued
    via prefill rather than spending the post-tool nudge on it; the
    ``not thinking`` guard on NUDGE keeps them mutually exclusive.
    """
    if visible or not limits.enabled:
        return RecoveryAction.COMPLETE

    thinking = has_thinking(response)

    # 只有思考的预填充：模型已推理，但没有生成正文。
    if thinking and prefill_retries < limits.thinking_prefill_max_retries:
        return RecoveryAction.PREFILL

    # 工具后空响应提示：排除只有思考的情况，因为上方已处理。
    if prev_had_tool_calls and not thinking and nudges_done < limits.post_tool_empty_max_nudges:
        return RecoveryAction.NUDGE

    # 兜底的普通重试。``prefill_exhausted`` 条件至关重要：部分模型始终填充推理字段，
    # 如果只在 ``not thinking`` 时允许重试，预填充用尽后所有推理模型都会被永久阻止重试。
    prefill_exhausted = thinking and prefill_retries >= limits.thinking_prefill_max_retries
    if empty_retries < limits.empty_content_max_retries and (not thinking or prefill_exhausted):
        return RecoveryAction.RETRY

    return RecoveryAction.COMPLETE
