"""Context Management engine.

One engine — :class:`ContextAssembler` — assembled by
:func:`build_context_engine` from a flat list of :class:`SegmentBuilder`
(seg1–5 + the Curator). The historical ``legacy`` / ``curator`` /
``default`` split has been collapsed.
"""

from pico.context_engine.assembler import ContextAssembler
from pico.context_engine.base import (
    AssembledPrefix,
    AssemblyContext,
    ContextEngine,
    Segment,
    SegmentBuilder,
)
from pico.context_engine.curator import TurnContext
from pico.context_engine.factory import build_context_engine
from pico.context_engine.history_trimmer import HistoryTrimmer

__all__ = [
    "AssembledPrefix",
    "AssemblyContext",
    "ContextAssembler",
    "ContextEngine",
    "HistoryTrimmer",
    "Segment",
    "SegmentBuilder",
    "TurnContext",
    "build_context_engine",
]
