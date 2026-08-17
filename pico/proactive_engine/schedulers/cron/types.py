"""Cron types."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CronSchedule:
    """Schedule definition for a cron job."""

    kind: Literal["at", "every", "cron"]
    # "at"：毫秒时间戳
    at_ms: int | None = None
    # "every"：毫秒间隔
    every_ms: int | None = None
    # "cron"：cron 表达式，如 "0 9 * * *"
    expr: str | None = None
    # cron 表达式的时区
    tz: str | None = None


@dataclass
class CronPayload:
    """What to do when the job runs."""

    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # 将响应投递到渠道
    deliver: bool = False
    channel: str | None = None  # 例如 "feishu"
    to: str | None = None  # 例如电话号码
    # 稳定的逻辑主题，用于更新现有任务，而不是为同一提醒创建近似重复的 schedule。
    topic_tag: str | None = None


@dataclass
class CronJobState:
    """Runtime state of a job."""

    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped", "expired", "cancelled", "incompatible"] | None = None
    last_error: str | None = None
    # claim 字段由在 run_due 中取得任务的进程设置，运行后清除。
    # 过期 claim（早于 CLAIM_TTL_MS）可被接管。
    claimed_by_pid: int | None = None
    claimed_at_ms: int | None = None
    # 自动衰减计数：没有用户活动介入时的连续触发次数。同一 channel/to 中任意
    # 用户来源消息都会将其重置为 0。silent-fires 守卫用它禁用 LLM 创建的
    # 失控周期任务，例如永久 every_seconds=3000。
    silent_fire_count: int = 0


@dataclass
class CronJob:
    """A scheduled job."""

    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False
    # 自动衰减上限：state.silent_fire_count 达到该值时自动禁用任务。
    # None 表示无限制，直到显式删除。默认 12 在两者间取平衡：按小时触发时
    # 约运行一天，之后判定“用户未参与”。
    silent_fire_limit: int | None = 12


@dataclass
class CronStore:
    """Persistent store for cron jobs."""

    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
