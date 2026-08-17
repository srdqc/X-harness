"""Reject code-class patches that carry no activation beacon.

Design section 3 (code class): every evolved code path must call
``activation_beacon(node_id)`` so Gate 1 can prove the path executed.
A diff with no beacon is unmonitorable — rejected before eval.
"""

from __future__ import annotations

# WHERE 按行为目标而非文件位置分类：直接编辑基准智能体代码（如终止策略）时，根据实际变更
# 分类为 loop_override、tool_override 或 context_override，因此没有单独的
# "benchmark_agent_code" 值。
CODE_CLASS_WHERES = {
    "tool_new",
    "loop_override",
    "context_override",
    "tool_override",
}
BEACON_TOKEN = "activation_beacon("


class MissingBeaconError(ValueError):
    """Raised when a code-class patch lacks an activation beacon."""

    pass


def assert_beacon_present(
    node_id: str,
    *,
    patch_where: str,
    diff_text: str,
) -> None:
    """Validate that a code-class patch contains an activation beacon.

    Args:
        node_id: The node ID (used in error messages).
        patch_where: The patch location (e.g. "loop_override", "skill").
        diff_text: The unified diff or code text to check.

    Raises:
        MissingBeaconError: If patch_where is a code class and diff_text
            contains no activation_beacon() call.
    """
    if patch_where not in CODE_CLASS_WHERES:
        return
    if BEACON_TOKEN not in diff_text:
        raise MissingBeaconError(
            f"{node_id}: {patch_where} patch contains no {BEACON_TOKEN}...) call - unmonitorable, rejected"
        )
