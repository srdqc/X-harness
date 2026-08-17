"""tui_rpc package - Python <-> Node JSON-RPC bridge for the Pico TUI.

Single source of truth for the contract lives in
``ui-tui/rpc-schema/openrpc.json``.  The Pydantic v2 models in
:mod:`pico.tui_rpc.models` are hand-written counterparts kept in sync via
``tests/test_rpc_schema_match.py``.
"""
