"""AgentHook abstraction for AgentLoop lifecycle.

Eval Engine and caller-supplied extensions share this contract across
user-inbound, iteration, Tool, and outbound phases.

Public surface:

- :class:`AgentHook`        — base class (all methods default to no-op).
- :class:`AgentHookContext` — per-turn state carried through the chain.
- :class:`HookDecision`     — what a hook chose: pass-through,
                              short-circuit, or content modification.
- :class:`CompositeHook`    — aggregate multiple hooks into one,
                              with short-circuit + content-chain
                              semantics and exception isolation.
"""

from pico.agent.hook.base import AgentHook, AgentHookContext, HookDecision
from pico.agent.hook.composite import CompositeHook

__all__ = [
    "AgentHook",
    "AgentHookContext",
    "HookDecision",
    "CompositeHook",
]
