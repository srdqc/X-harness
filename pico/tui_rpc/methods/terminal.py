"""``terminal.resize`` RPC handler — record cols, return ok.

ui-tui's ``useMainApp.ts:426`` calls ``terminal.resize`` with the new
``{cols, rows}`` payload whenever Ink observes a SIGWINCH; the call is
fire-and-forget. We need a handler that:

  1. Never raises (so the SIGWINCH burst doesn't spam errors), and
  2. Records the latest dimensions for the active TUI session.

The recorded state is module-level (a single TUI subprocess has exactly one
terminal, so a singleton is correct).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pico.tui_rpc.dispatcher import Dispatcher


# 模块级的最新已知终端尺寸。``None`` 表示尚未观察到 resize 事件；
# 此时调用方应回退到 ``shutil.get_terminal_size()`` 或合理默认值（80 列）。
_LATEST_COLS: int | None = None
_LATEST_ROWS: int | None = None


def get_latest_cols() -> int | None:
    """Return the most recently reported terminal column count, or ``None``.

    The TUI runtime can use this value when sizing terminal output.
    """
    return _LATEST_COLS


def get_latest_rows() -> int | None:
    """Return the most recently reported terminal row count, or ``None``."""
    return _LATEST_ROWS


def _coerce_dim(value: Any) -> int | None:
    """Return ``value`` as a positive int, else ``None``."""
    if isinstance(value, bool):
        # bool 是 int 的子类，需显式拒绝。
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


async def terminal_resize(params: dict) -> dict:
    """``terminal.resize`` — record dimensions, return ``{ok: true}``.

    Accepts ``{cols, rows}`` (both optional positive ints). Anything else is
    silently coerced to a no-op record — we never raise here because the
    upstream SIGWINCH burst would otherwise flood error frames.
    """
    global _LATEST_COLS, _LATEST_ROWS
    if isinstance(params, dict):
        cols = _coerce_dim(params.get("cols"))
        rows = _coerce_dim(params.get("rows"))
        if cols is not None:
            _LATEST_COLS = cols
        if rows is not None:
            _LATEST_ROWS = rows
    return {"ok": True}


def register_terminal_methods(dispatcher: "Dispatcher") -> None:
    """Register ``terminal.resize`` on a dispatcher instance."""
    dispatcher.register("terminal.resize", terminal_resize)


__all__ = [
    "terminal_resize",
    "register_terminal_methods",
    "get_latest_cols",
    "get_latest_rows",
]
