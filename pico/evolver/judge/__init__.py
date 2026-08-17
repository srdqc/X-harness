"""LLM-judge subsystem for the evolver.

Reads compressed trajectories, classifies failures as L1/L2/L3
(spec §3), and proposes structured (WHERE, WHY) patches for L2/L3 cases
(spec §12.4-§12.5).

Public surface:

- ``IssueType`` / ``PatchWhere`` / ``PatchWhy`` / ``ActionKind`` — enums
- ``JudgeAction`` / ``JudgeResult`` — parsed analysis dataclasses
- ``build_judge_messages`` — assemble (system, user) for one LLM call
- ``parse_judge_output`` — turn LLM raw text into a validated JudgeResult
- ``JudgeParseError`` — raised on malformed output

The LLM client itself is in ``pico.evolver.judge.llm_client`` (B3,
written separately).
"""

from .llm_client import (
    JudgeLLM,
    JudgeLLMBackend,
    JudgeLLMConfig,
    LitellmBackend,
    MockBackend,
    Mode,
    OpenRouterBackend,
    TrajectoryFormat,
    build_backend,
    build_judge_llm,
)
from .parser import JudgeParseError, parse_judge_output
from .prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    WHERE_DESCRIPTIONS,
    WHY_DESCRIPTIONS,
    build_judge_messages,
)
from .schema import (
    ActionKind,
    IssueType,
    JudgeAction,
    JudgeResult,
    PatchWhere,
    PatchWhy,
    ProposedComponent,
)

__all__ = [
    # 枚举
    "ActionKind",
    "IssueType",
    "PatchWhere",
    "PatchWhy",
    # 数据类
    "JudgeAction",
    "JudgeResult",
    "ProposedComponent",
    # 提示构建
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_USER_TEMPLATE",
    "WHERE_DESCRIPTIONS",
    "WHY_DESCRIPTIONS",
    "build_judge_messages",
    # 解析
    "JudgeParseError",
    "parse_judge_output",
    # LLM 客户端（B3）
    "JudgeLLM",
    "JudgeLLMBackend",
    "JudgeLLMConfig",
    "LitellmBackend",
    "MockBackend",
    "Mode",
    "OpenRouterBackend",
    "TrajectoryFormat",
    "build_backend",
    "build_judge_llm",
]
