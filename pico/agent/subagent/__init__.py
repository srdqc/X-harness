"""SubagentManager — spawns child AgentLoops for delegated tasks.

Implementation lives in ``manager.py``.

External callers should keep using:

    from pico.agent.subagent import SubagentManager
"""

from pico.agent.subagent.manager import SubagentManager

__all__ = ["SubagentManager"]
