"""Segment 1 - ``# Pico`` identity / runtime. Host-owned."""

from __future__ import annotations

from pathlib import Path

from pico.context_engine.base import AssemblyContext, Segment
from pico.context_engine.segments import render


class IdentitySegmentBuilder:
    name = "identity"
    order = 1
    needs_prefix = False

    def __init__(self, workspace: Path, state: Path | None = None) -> None:
        self._workspace = workspace
        self._state = state

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        return Segment(text=render.identity_text(self._workspace, self._state))
