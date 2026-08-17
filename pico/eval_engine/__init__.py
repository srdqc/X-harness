"""Eval Engine — L3 cognition-coord task judge.

Provides three AgentHook implementations that ride
on AgentLoop's lifecycle phases to answer three orthogonal questions:

- ``BeforeIterationHook``  — "should we even start the next iteration?"
                              (token budget / pruning)
- ``ToolAuditHook``         — "is this tool call safe to execute?"
                              (deny-list / approval workflow stub)
- ``AfterIterationHook``    — "did this turn complete successfully?"
                              (LLM judge over the final response;
                              records the verdict through the memory adapter)

All three are **off by default** (``EvalEngineConfig.enabled = False``).
Mounting them onto AgentLoop happens via the CLI stack — see
``cli/_eval_stack.py`` for the wire-up.

Layout:
    eval_engine/
      ├── config.py              Pydantic ``EvalEngineConfig``
      ├── engine.py              ``EvalEngine`` orchestrator
      ├── hooks/
      │   ├── before_iteration_hook.py
      │   ├── tool_audit_hook.py
      │   └── after_iteration_hook.py
      ├── judge/
      │   └── judge.py           LLM judge invocation
      ├── adapter/
      │   └── adapter.py         MemoryEngine write-back
      └── prompts/
          ├── task_completion.py
          └── tool_safety.py
"""

from pico.eval_engine.config import EvalEngineConfig
from pico.eval_engine.engine import EvalEngine
from pico.eval_engine.hooks.after_iteration_hook import AfterIterationHook
from pico.eval_engine.hooks.before_iteration_hook import BeforeIterationHook
from pico.eval_engine.hooks.tool_audit_hook import ToolAuditHook
from pico.eval_engine.judge.judge import EvalJudge, JudgeVerdict

__all__ = [
    "EvalEngine",
    "EvalEngineConfig",
    "BeforeIterationHook",
    "ToolAuditHook",
    "AfterIterationHook",
    "EvalJudge",
    "JudgeVerdict",
]
