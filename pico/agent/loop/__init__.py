"""AgentLoop - the Pico L2 ReAct executor.

The full ``AgentLoop`` implementation lives in ``main.py``. The package
shape is in place so the file can later be split into ``main.py`` /
``dispatch.py`` / ``runner.py`` without further import churn.

External callers should keep using:

    from pico.agent.loop import AgentLoop

which re-exports through here.
"""

from pico.agent.loop.main import AgentLoop, TurnOutcome

__all__ = ["AgentLoop", "TurnOutcome"]
