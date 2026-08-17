"""EvalEngine orchestrator.

Holds a config and constructs the three AgentHook instances plus the
judge + adapter dependencies. Exposes a single :meth:`hooks` accessor
that returns the three hooks in a stable order, so an Eval-aware
CLI stack can ``CompositeHook.extend(engine.hooks())`` without
re-implementing the wiring.

Designed so a caller without an LLM provider or MemoryEngine can
construct a degraded EvalEngine — useful for tests that only want
to exercise the deterministic deny-list path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pico.eval_engine.adapter.adapter import EvalAdapter
from pico.eval_engine.config import EvalEngineConfig
from pico.eval_engine.hooks.after_iteration_hook import AfterIterationHook
from pico.eval_engine.hooks.before_iteration_hook import BeforeIterationHook
from pico.eval_engine.hooks.tool_audit_hook import ToolAuditHook
from pico.eval_engine.judge.judge import EvalJudge

if TYPE_CHECKING:
    from pico.agent.hook import AgentHook
    from pico.memory_engine.consolidate.consolidator import MemoryStore
    from pico.providers.base import LLMProvider


class EvalEngine:
    """Aggregates the three Eval Engine hooks behind a single factory.

    Phase B-3: the ``memory`` arg was re-typed from the (deleted)
    ``MemoryEngine`` facade to :class:`MemoryStore` since the only
    method the adapter uses is ``append_history``.
    """

    def __init__(
        self,
        config: EvalEngineConfig | None = None,
        *,
        memory: "MemoryStore | None" = None,
        provider: "LLMProvider | None" = None,
    ) -> None:
        self._config = config or EvalEngineConfig()

        # Judge 需要 Provider；若未提供（例如测试框架或禁用配置的部署），
        # 则替换为始终返回 unknown 的桩，使 after-iteration hook 保持静默。
        if provider is not None:
            self._judge: EvalJudge = EvalJudge(
                provider,
                model=self._config.judge_model,
                timeout_seconds=self._config.judge_timeout_seconds,
            )
        else:
            self._judge = _NoopJudge()  # type: ignore[assignment]

        self._adapter: EvalAdapter | None = EvalAdapter(memory) if memory is not None else None

        self._before_iteration = BeforeIterationHook(self._config)
        self._tool_audit = ToolAuditHook(self._config)
        self._after_iteration = (
            AfterIterationHook(self._config, self._judge, self._adapter) if self._adapter is not None else _NoopHook()  # type: ignore[assignment]
        )

    @property
    def config(self) -> EvalEngineConfig:
        return self._config

    def hooks(self) -> list["AgentHook"]:
        """Return the three hooks in canonical iteration order."""
        return [
            self._before_iteration,
            self._tool_audit,
            self._after_iteration,
        ]


# ---------------------------------------------------------------------------
# Engine 在没有 Provider / MemoryEngine 时使用的桩回退。
# 让 ``EvalEngine.hooks()`` 仍返回真实 AgentHook 实例，确保下游
# CompositeHook 使用时满足类型契约。
# ---------------------------------------------------------------------------


from pico.agent.hook.base import AgentHook, AgentHookContext, HookDecision
from pico.eval_engine.judge.judge import JudgeVerdict


class _NoopJudge:
    """Drop-in judge used when no LLM provider is supplied. Always
    returns ``JudgeVerdict.unknown`` so the AfterIterationHook stays
    quiet."""

    async def judge(
        self,
        user_goal: str,
        final_response: str,
        messages=None,
    ) -> JudgeVerdict:
        return JudgeVerdict.unknown


class _NoopHook(AgentHook):
    """Fallback hook used when no MemoryEngine is wired into the
    EvalEngine — every phase is pass-through. Distinct from a plain
    ``AgentHook()`` only so debug logs identify it clearly."""

    @property
    def name(self) -> str:
        return "EvalNoopHook"

    async def after_iteration(self, ctx: AgentHookContext) -> HookDecision:
        return HookDecision()


__all__ = ["EvalEngine"]
