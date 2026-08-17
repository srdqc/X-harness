"""ContextBuilder — assembles system prompt + history for AgentLoop.

Implementation lives in ``builder.py``.

External callers should keep using:

    from pico.agent.context import ContextBuilder
"""

from pico.agent.context.builder import ContextBuilder

__all__ = ["ContextBuilder"]
