"""Data types for the EcoClaw-style model router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ── 基准数据 ───────────────────────────────────────────────────────────


@dataclass
class ModelTaskScore:
    task_id: str
    score: float  # 0 到 100 的百分比
    max_score: float


@dataclass
class ModelBenchmark:
    model: str  # PinchBench 模型 ID，例如 "anthropic/claude-sonnet-4"
    provider: str
    overall_score: float
    speed: float | None  # 平均执行时间（秒），由 PinchBench 记录
    cost: float  # 完整运行 23 项基准任务的总成本（美元）
    task_scores: list[ModelTaskScore]
    submission_id: str


# ── 路由配置档 ─────────────────────────────────────────────────────────

RoutingProfileName = Literal["best", "balanced", "eco"]


@dataclass(frozen=True)
class RoutingProfile:
    quality_weight: float
    cost_weight: float


# ── 分类 ───────────────────────────────────────────────────────────────

TaskCategory = Literal[
    "sanity",
    "calendar",
    "stock",
    "blog",
    "tool_use",
    "summary",
    "events",
    "email",
    "memory",
    "files",
    "workflow",
    "clawdhub",
    "skill_search",
    "image_gen",
    "humanizer",
    "daily_summary",
    "email_triage",
    "email_search",
    "market_research",
    "spreadsheet",
    "eli5_pdf",
    "comprehension",
    "second_brain",
]


@dataclass
class ClassificationResult:
    category: TaskCategory
    similarity: float


# ── 选择 ───────────────────────────────────────────────────────────────


@dataclass
class ModelScore:
    model: str
    provider: str
    task_score: float
    cost_score: float
    composite_score: float


@dataclass
class SelectionResult:
    primary: ModelScore
    fallbacks: list[ModelScore]
    category: TaskCategory
    profile: RoutingProfileName
