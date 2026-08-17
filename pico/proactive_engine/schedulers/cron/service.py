"""Cron service for scheduling agent tasks."""

import asyncio
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Iterator

from loguru import logger

from pico.proactive_engine.schedulers.cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore
from pico.utils.portable_lock import file_lock

# 过期 claim 的 TTL：超过该时长后其他进程可以接管，原进程很可能在任务中途崩溃。
_CLAIM_TTL_MS = 30 * 60 * 1000

# 限制下一次唤醒前的最长睡眠时间，保证 run_due 至少按此频率运行。
# run_due 会在 mtime 变化时重载 store，从而发现同级进程写入 jobs.json 的任务。
# 没有该上限时，等待遥远唤醒时间的 gateway 会错过 REPL 新增的更早任务。
_MAX_WAKE_INTERVAL_S = 30.0


def _expect_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a boolean")
    return value


def _expect_int(value: Any, field: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if type(value) is not int:
        suffix = " or null" if allow_none else ""
        raise TypeError(f"{field} must be an integer{suffix}")
    return value


def _expect_str(value: Any, field: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        suffix = " or null" if allow_none else ""
        raise TypeError(f"{field} must be a string{suffix}")
    return value


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """Compute next run time in ms."""
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter

            # 使用调用方提供的参考时间，确保调度结果确定。
            base_time = now_ms / 1000
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_time, tz=tz)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception:
            return None

    return None


def _validate_schedule_for_add(schedule: CronSchedule, now_ms: int) -> None:
    """Validate schedule fields that would otherwise create non-runnable jobs."""
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")

    if schedule.kind == "cron" and schedule.tz:
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(schedule.tz)
        except Exception:
            raise ValueError(f"unknown timezone '{schedule.tz}'") from None

        # 没有下次运行时间的 schedule 若被保存，会成为永不触发的任务，向调用方
        # 造成假成功。_compute_next_run 是“可运行”的唯一事实来源，因此此处拒绝
        # 任何被它映射为 None 的类型。
    if _compute_next_run(schedule, now_ms) is None:
        if schedule.kind == "at":
            raise ValueError("at time is in the past")
        if schedule.kind == "every":
            raise ValueError("every_seconds must be positive")
        if schedule.kind == "cron":
            raise ValueError(f"invalid cron expression '{schedule.expr}'")
        raise ValueError(f"schedule kind '{schedule.kind}' is not runnable")


class CronService:
    """Service for managing and executing scheduled jobs."""

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
        *,
        allowed_channels: set[str] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        """``allowed_channels`` restricts which jobs this service will claim.

        Set to e.g. ``{"cli"}`` in REPL mode so REPL doesn't steal Feishu /
        Telegram reminders that gateway should deliver. ``None`` (default)
        means any channel — use that in gateway where ChannelManager can
        route replies to any configured channel.

        Jobs with empty/None ``payload.channel`` are always claimable —
        they predate the channel attribution field.
        """
        self.store_path = store_path
        # 相邻文件用于 advisory lock；数据文件原子重命名后它仍存在，
        # 让并发进程协调 run_due。
        self.lock_path = store_path.with_suffix(store_path.suffix + ".lock")
        self.on_job = on_job
        self.allowed_channels = allowed_channels
        self._store: CronStore | None = None
        self._incompatible_records: dict[int, Any] = {}
        self._store_metadata: dict[str, Any] = {}
        # 使用纳秒精度：浮点 st_mtime 会把间隔约 238 纳秒的写入合并
        # 合并成一个值，避免外部重写后仍提供过期缓存。
        self._last_mtime: int = 0
        self._timer_task: asyncio.Task | None = None
        self._running = False
        # 为 benchmark harness（longrun）提供可选的伪时钟注入。设置后，内部
        # 所有时间读取都经过该 callable，使新任务的 next_run_at_ms 对齐模拟时间，
        # 而不是真实 wall-clock。
        self._now_fn = now_fn

    def _now_ms(self) -> int:
        """Return current time in ms, honouring fake-clock injection."""
        if self._now_fn is not None:
            return int(self._now_fn().timestamp() * 1000)
        return int(time.time() * 1000)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the cross-platform lock on the jobs-file sibling."""
        with file_lock(self.lock_path):
            yield

    def _load_store(self) -> CronStore:
        """Load jobs from disk. Reloads automatically if file was modified externally."""
        if self._store and self.store_path.exists():
            mtime = self.store_path.stat().st_mtime_ns
            if mtime != self._last_mtime:
                logger.info("Cron: jobs.json modified externally, reloading")
                self._store = None
        if self._store:
            return self._store

        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise TypeError("cron store must be an object")
                raw_jobs = data.get("jobs", [])
                if not isinstance(raw_jobs, list):
                    raise TypeError("cron store jobs must be an array")
                store_version = _expect_int(data.get("version", 1), "store.version")
                self._incompatible_records = {}
                self._store_metadata = {key: value for key, value in data.items() if key not in {"version", "jobs"}}
                jobs = []
                for index, raw_job in enumerate(raw_jobs):
                    try:
                        if store_version != 1:
                            raise ValueError(f"unsupported store version '{store_version}'")
                        jobs.append(self._deserialize_job(raw_job))
                    except (KeyError, TypeError, ValueError) as error:
                        logger.warning("Cron: incompatible job at index {}: {}", index, error)
                        incompatible = self._incompatible_job(raw_job, index, str(error))
                        self._incompatible_records[id(incompatible)] = raw_job
                        jobs.append(incompatible)
                self._store = CronStore(version=store_version, jobs=jobs)
            except Exception as e:
                logger.warning("Failed to load cron store: {}", e)
                self._incompatible_records = {}
                self._store_metadata = {}
                self._store = CronStore()
        else:
            self._incompatible_records = {}
            self._store_metadata = {}
            self._store = CronStore()

        return self._store

    @staticmethod
    def _deserialize_job(raw_job: Any) -> CronJob:
        if not isinstance(raw_job, dict):
            raise TypeError("job record must be an object")
        job_id = _expect_str(raw_job["id"], "job.id")
        name = _expect_str(raw_job["name"], "job.name")
        enabled = _expect_bool(raw_job.get("enabled", True), "job.enabled")
        schedule_data = raw_job["schedule"]
        payload_data = raw_job["payload"]
        if not isinstance(schedule_data, dict):
            raise TypeError("schedule must be an object")
        if not isinstance(payload_data, dict):
            raise TypeError("payload must be an object")
        schedule_kind = _expect_str(schedule_data["kind"], "schedule.kind")
        if schedule_kind not in {"at", "every", "cron"}:
            raise ValueError(f"unsupported schedule kind '{schedule_kind}'")
        at_ms = _expect_int(schedule_data.get("atMs"), "schedule.atMs", allow_none=True)
        every_ms = _expect_int(schedule_data.get("everyMs"), "schedule.everyMs", allow_none=True)
        expr = _expect_str(schedule_data.get("expr"), "schedule.expr", allow_none=True)
        tz = _expect_str(schedule_data.get("tz"), "schedule.tz", allow_none=True)
        payload_kind = _expect_str(payload_data.get("kind", "agent_turn"), "payload.kind")
        if payload_kind not in {"agent_turn", "system_event"}:
            raise ValueError(f"unsupported payload kind '{payload_kind}'")
        message = _expect_str(payload_data.get("message", ""), "payload.message")
        deliver = _expect_bool(payload_data.get("deliver", False), "payload.deliver")
        channel = _expect_str(payload_data.get("channel"), "payload.channel", allow_none=True)
        to = _expect_str(payload_data.get("to"), "payload.to", allow_none=True)
        topic_tag = _expect_str(payload_data.get("topicTag"), "payload.topicTag", allow_none=True)
        state_data = raw_job.get("state", {})
        if not isinstance(state_data, dict):
            raise TypeError("state must be an object")
        next_run_at_ms = _expect_int(state_data.get("nextRunAtMs"), "state.nextRunAtMs", allow_none=True)
        last_run_at_ms = _expect_int(state_data.get("lastRunAtMs"), "state.lastRunAtMs", allow_none=True)
        last_status = _expect_str(state_data.get("lastStatus"), "state.lastStatus", allow_none=True)
        if last_status not in {None, "ok", "error", "skipped", "expired", "cancelled", "incompatible"}:
            raise ValueError(f"unsupported job status '{last_status}'")
        last_error = _expect_str(state_data.get("lastError"), "state.lastError", allow_none=True)
        claimed_by_pid = _expect_int(state_data.get("claimedByPid"), "state.claimedByPid", allow_none=True)
        claimed_at_ms = _expect_int(state_data.get("claimedAtMs"), "state.claimedAtMs", allow_none=True)
        silent_fire_count = _expect_int(state_data.get("silentFireCount", 0), "state.silentFireCount")
        created_at_ms = _expect_int(raw_job.get("createdAtMs", 0), "job.createdAtMs")
        updated_at_ms = _expect_int(raw_job.get("updatedAtMs", 0), "job.updatedAtMs")
        delete_after_run = _expect_bool(raw_job.get("deleteAfterRun", False), "job.deleteAfterRun")
        silent_fire_limit = _expect_int(raw_job.get("silentFireLimit", 12), "job.silentFireLimit", allow_none=True)
        return CronJob(
            id=job_id,
            name=name,
            enabled=enabled,
            schedule=CronSchedule(
                kind=schedule_kind,
                at_ms=at_ms,
                every_ms=every_ms,
                expr=expr,
                tz=tz,
            ),
            payload=CronPayload(
                kind=payload_kind,
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
                topic_tag=topic_tag,
            ),
            state=CronJobState(
                next_run_at_ms=next_run_at_ms,
                last_run_at_ms=last_run_at_ms,
                last_status=last_status,
                last_error=last_error,
                claimed_by_pid=claimed_by_pid,
                claimed_at_ms=claimed_at_ms,
                silent_fire_count=silent_fire_count,
            ),
            created_at_ms=created_at_ms,
            updated_at_ms=updated_at_ms,
            delete_after_run=delete_after_run,
            silent_fire_limit=silent_fire_limit,
        )

    @staticmethod
    def _incompatible_job(raw_job: Any, index: int, error: str) -> CronJob:
        record = raw_job if isinstance(raw_job, dict) else {}
        payload = record.get("payload")
        payload_message = payload.get("message", "") if isinstance(payload, dict) else ""
        job_id = record.get("id")
        name = record.get("name")
        return CronJob(
            id=job_id if isinstance(job_id, str) and job_id else f"incompatible-{index}",
            name=name if isinstance(name, str) and name else "Incompatible cron job",
            enabled=False,
            schedule=CronSchedule(kind="at"),
            payload=CronPayload(message=payload_message if isinstance(payload_message, str) else ""),
            state=CronJobState(last_status="incompatible", last_error=error),
            created_at_ms=record.get("createdAtMs", 0),
            updated_at_ms=record.get("updatedAtMs", 0),
            delete_after_run=False,
        )

    def _save_store(self) -> None:
        """Save jobs to disk."""
        if not self._store:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            **self._store_metadata,
            "version": self._store.version,
            "jobs": [self._serialize_job(job) for job in self._store.jobs],
        }

        # 通过临时文件 + rename 原子写入，避免并发读取方看到只刷新了一部分的文件。
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.store_path)
        self._last_mtime = self.store_path.stat().st_mtime_ns

    def _serialize_job(self, job: CronJob) -> Any:
        if id(job) in self._incompatible_records:
            return self._incompatible_records[id(job)]
        return {
            "id": job.id,
            "name": job.name,
            "enabled": job.enabled,
            "schedule": {
                "kind": job.schedule.kind,
                "atMs": job.schedule.at_ms,
                "everyMs": job.schedule.every_ms,
                "expr": job.schedule.expr,
                "tz": job.schedule.tz,
            },
            "payload": {
                "kind": job.payload.kind,
                "message": job.payload.message,
                "deliver": job.payload.deliver,
                "channel": job.payload.channel,
                "to": job.payload.to,
                "topicTag": job.payload.topic_tag,
            },
            "state": {
                "nextRunAtMs": job.state.next_run_at_ms,
                "lastRunAtMs": job.state.last_run_at_ms,
                "lastStatus": job.state.last_status,
                "lastError": job.state.last_error,
                "claimedByPid": job.state.claimed_by_pid,
                "claimedAtMs": job.state.claimed_at_ms,
                "silentFireCount": job.state.silent_fire_count,
            },
            "createdAtMs": job.created_at_ms,
            "updatedAtMs": job.updated_at_ms,
            "deleteAfterRun": job.delete_after_run,
            "silentFireLimit": job.silent_fire_limit,
        }

    async def start(self) -> None:
        """Start the cron service."""
        self._running = True
        self._load_store()
        self._recompute_next_runs()
        self._save_store()
        self._arm_timer()
        logger.info("Cron service started with {} jobs", len(self._store.jobs if self._store else []))

    def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    def _recompute_next_runs(self) -> None:
        """Recompute next run times for all enabled jobs.

        Jobs keep one persisted pending fire across restarts. A missed
        recurring fire runs once and then advances from the recovery time
        instead of backfilling every missed interval.
        """
        if not self._store:
            return
        now = self._now_ms()
        for job in self._store.jobs:
            if not job.enabled:
                continue
            if job.schedule.kind == "at" and job.state.last_status in {
                "ok",
                "error",
                "expired",
                "incompatible",
            }:
                job.enabled = False
                job.state.next_run_at_ms = None
                continue
            if job.schedule.kind == "at" and job.schedule.at_ms is not None and job.schedule.at_ms > 0:
                job.state.next_run_at_ms = job.schedule.at_ms
                continue
            next_run = _compute_next_run(job.schedule, now)
            if next_run is None:
                job.state.next_run_at_ms = None
                job.enabled = False
                job.state.last_status = "expired"
                job.state.last_error = "schedule has no future fire"
                job.updated_at_ms = now
                logger.warning("Cron: expired job '{}' ({}): {}", job.name, job.id, job.state.last_error)
            elif job.state.next_run_at_ms is None:
                job.state.next_run_at_ms = next_run

    def _get_next_wake_ms(self) -> int | None:
        """Get the earliest next run time across all jobs."""
        if not self._store:
            return None
        return min(
            (j.state.next_run_at_ms for j in self._store.jobs if j.enabled and j.state.next_run_at_ms),
            default=None,
        )

    def _arm_timer(self) -> None:
        """Schedule the next timer tick.

        Always sleeps at most ``_MAX_WAKE_INTERVAL_S`` so a peer process's
        write to jobs.json (e.g. a new reminder from REPL while gateway is
        running) gets picked up within that window — run_due reloads on
        mtime change.
        """
        if self._timer_task:
            self._timer_task.cancel()

        if not self._running:
            return

        next_wake = self._get_next_wake_ms()
        if next_wake:
            delay_s = max(0.0, (next_wake - self._now_ms()) / 1000)
        else:
            # 没有待执行任务时仍需轮询新写入。
            delay_s = _MAX_WAKE_INTERVAL_S
        delay_s = min(delay_s, _MAX_WAKE_INTERVAL_S)

        async def tick():
            await asyncio.sleep(delay_s)
            if self._running:
                await self.run_due()

        self._timer_task = asyncio.create_task(tick())

    async def run_due(self) -> None:
        """Claim and run jobs that are due.

        Claim phase (under exclusive lock): reload from disk, pick due jobs
        not already claimed by a live peer, stamp them with this pid+now,
        save. Execution phase (lock released): run each claimed job; then
        reacquire the lock to write post-run state and clear the claim.
        """
        my_pid = os.getpid()
        with self._locked():
            # 强制重读：同级进程可能已在此期间修改数据。
            self._store = None
            self._load_store()
            if not self._store:
                return

            now = self._now_ms()
            my_jobs: list[CronJob] = []
            for j in self._store.jobs:
                if not (j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms):
                    continue
                # 渠道路由：调用方设置 allow-list 后，只 claim 其中渠道的任务。
                # 为向后兼容，在 channel attribution 出现前创建的任务
                # （channel 为空或 None）仍可被任意进程 claim。
                if self.allowed_channels is not None and j.payload.channel:
                    if j.payload.channel not in self.allowed_channels:
                        continue
                # 已被存活的同级进程持有时跳过。
                cb = j.state.claimed_by_pid
                ca = j.state.claimed_at_ms
                if cb is not None and ca is not None and (now - ca) < _CLAIM_TTL_MS:
                    continue
                j.state.claimed_by_pid = my_pid
                j.state.claimed_at_ms = now
                my_jobs.append(j)
            if my_jobs:
                self._save_store()

        try:
            for job in my_jobs:
                try:
                    await self._execute_job(job)
                finally:
                    self._persist_job_result(job, my_pid, now)
        finally:
            self._release_claims([job.id for job in my_jobs], my_pid, now)
            self._arm_timer()

    def _persist_job_result(self, job: CronJob, claimed_by_pid: int, claimed_at_ms: int) -> None:
        with self._locked():
            self._store = None
            self._load_store()
            if self._store is None:
                return
            owns_claim = False
            for persisted in self._store.jobs:
                if (
                    persisted.id != job.id
                    or persisted.state.claimed_by_pid != claimed_by_pid
                    or persisted.state.claimed_at_ms != claimed_at_ms
                ):
                    continue
                owns_claim = True
                persisted.state.claimed_by_pid = None
                persisted.state.claimed_at_ms = None
                persisted.state.last_run_at_ms = job.state.last_run_at_ms
                persisted.state.last_status = job.state.last_status
                persisted.state.last_error = job.state.last_error
                persisted.state.next_run_at_ms = job.state.next_run_at_ms
                persisted.enabled = job.enabled
                persisted.updated_at_ms = job.updated_at_ms
                break
            if not owns_claim:
                return
            if job.schedule.kind == "at" and job.delete_after_run and job.state.last_status != "cancelled":
                self._store.jobs = [persisted for persisted in self._store.jobs if persisted.id != job.id]
            self._save_store()

    def _release_claims(self, job_ids: list[str], claimed_by_pid: int, claimed_at_ms: int) -> None:
        if not job_ids:
            return
        with self._locked():
            self._store = None
            self._load_store()
            if self._store is None:
                return
            changed = False
            for job in self._store.jobs:
                if (
                    job.id in job_ids
                    and job.state.claimed_by_pid == claimed_by_pid
                    and job.state.claimed_at_ms == claimed_at_ms
                ):
                    job.state.claimed_by_pid = None
                    job.state.claimed_at_ms = None
                    changed = True
            if changed:
                self._save_store()

    async def _execute_job(self, job: CronJob) -> None:
        """Execute a single job."""
        start_ms = self._now_ms()
        logger.info("Cron: executing job '{}' ({})", job.name, job.id)

        try:
            if self.on_job:
                await self.on_job(job)

            job.state.last_status = "ok"
            job.state.last_error = None
            logger.info("Cron: job '{}' completed", job.name)

        except asyncio.CancelledError:
            job.state.last_run_at_ms = start_ms
            job.state.last_status = "cancelled"
            job.state.last_error = "execution cancelled"
            job.updated_at_ms = self._now_ms()
            logger.warning("Cron: job '{}' cancelled", job.name)
            raise
        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.error("Cron: job '{}' failed: {}", job.name, e)

        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = self._now_ms()

        # 处理一次性任务。
        if job.schedule.kind == "at":
            if job.delete_after_run:
                job.state.next_run_at_ms = None
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        elif job.enabled:
            job.state.next_run_at_ms = _compute_next_run(job.schedule, self._now_ms())
        else:
            # 这是禁用状态下被强制触发的周期任务（CLI `cron run --force`）。
            # 不推进 next_run_at_ms：任务仍处于禁用状态，未来的 next-run 与
            # enabled=False 同时出现会误导 `cron list` 输出。
            job.state.next_run_at_ms = None

    # ========== 公共 API ==========

    def record_silent_fire(self, job_id: str) -> bool:
        """Increment silent_fire_count for a job; auto-disable when it
        crosses silent_fire_limit. Called by harness/dispatch path right
        after a cron fire is delivered. Returns True if the job was
        auto-disabled this call."""
        with self._locked():
            self._store = None
            store = self._load_store()
            for j in store.jobs:
                if j.id != job_id:
                    continue
                j.state.silent_fire_count += 1
                limit = j.silent_fire_limit
                disabled = False
                if limit is not None and limit > 0 and j.state.silent_fire_count >= limit:
                    j.enabled = False
                    j.state.next_run_at_ms = None
                    disabled = True
                    logger.warning(
                        "Cron: auto-disabled job '{}' ({}) — {} silent fires without user activity (limit={})",
                        j.name,
                        j.id,
                        j.state.silent_fire_count,
                        limit,
                    )
                self._save_store()
                return disabled
            return False

    def notify_user_active(self, channel: str | None = None, to: str | None = None) -> int:
        """Reset silent_fire_count for jobs matching (channel, to) — call
        whenever a genuine user-originated message arrives so recently-
        firing crons don't decay toward auto-disable. None matches all.
        Returns count of jobs whose state was reset."""
        reset = 0
        with self._locked():
            self._store = None
            store = self._load_store()
            for j in store.jobs:
                if not j.enabled or j.state.silent_fire_count == 0:
                    continue
                if channel is not None and j.payload.channel != channel:
                    continue
                if to is not None and j.payload.to != to:
                    continue
                j.state.silent_fire_count = 0
                reset += 1
            if reset > 0:
                self._save_store()
        return reset

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """List all jobs."""
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float("inf"))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
        topic_tag: str | None = None,
    ) -> CronJob:
        """Add a new job, or update an existing job with the same
        (schedule, channel, to) triple — agents often re-register the
        same recurring reminder with slightly different wording across
        conversations; without dedup the user gets N near-identical
        fires per scheduled tick.

        Two cross-kind dedup layers also apply (in order):

        1. **Message-equal dedup**: if any existing enabled job for the
           same (channel, to) has a *byte-identical* ``payload.message``,
           return it. Catches the case where the LLM creates the same
           reminder N times across the simulation horizon (e.g. a
           medication-reminder string appearing as both an ``at`` shot
           today and a ``cron_expr`` recurring tomorrow — identical
           text, different fire times).

        2. **Time-window dedup**: if any existing enabled job for the
           same (channel, to) is scheduled to fire within 15 minutes of
           this new schedule's next fire (regardless of schedule kind),
           return it. Catches the case where the LLM creates both a
           recurring ``cron_expr`` AND a same-day ``at`` shot for the
           same intent (e.g. "daily 8:00 take meds" + "today 8:00 take
           meds").
        """
        # 校验与存储共用同一个 ``now`` 快照。校验谓词和存储的 next_run 必须对
        # “现在”达成一致，否则临界 ``at``（at ≈ now）可能通过校验却保存为
        # next_run=None。锁外取值，使无效 schedule 无需争锁即可快速失败。
        now = self._now_ms()
        _validate_schedule_for_add(schedule, now)
        with self._locked():
            # 在锁内重载，避免覆盖并发写入方新增的任务。
            self._store = None
            store = self._load_store()
            if store.version != 1:
                raise ValueError(f"unsupported cron store version '{store.version}'")

            # L7：topic_tag 去重最严格，因此最先执行。新请求带 topic_tag 时，
            # 相同 (channel, to, topic_tag) 的任何已启用任务都视为重复。这能捕获
            # 护理场景中的失败模式：LLM 创建近乎相同的用药提醒 cron，只在时间
            # （11:20 与 11:30）或措辞上略有差异；消息全等去重和 15 分钟窗口去重
            # 都会漏掉。topic_tag 就是“提醒主题”的身份，因此相同 topic_tag 的
            # 两个 cron 按定义属于同一逻辑提醒。就地更新现有任务的消息/schedule，
            # 而不是创建并行任务。
            if topic_tag:
                for j in store.jobs:
                    if not j.enabled:
                        continue
                    if j.payload.channel != channel or j.payload.to != to:
                        continue
                    if j.payload.topic_tag != topic_tag:
                        continue
                    logger.info(
                        "Cron: topic_tag dedup — existing job '{}' ({}) "
                        "has topic_tag='{}'; updating message + schedule "
                        "in place (kinds={}/{})",
                        j.name,
                        j.id,
                        topic_tag,
                        j.schedule.kind,
                        schedule.kind,
                    )
                    j.payload.message = message
                    j.payload.deliver = deliver
                    j.name = name
                    j.schedule = schedule
                    j.state.next_run_at_ms = _compute_next_run(schedule, now)
                    j.updated_at_ms = now
                    self._save_store()
                    self._arm_timer()
                    return j

            # 消息全等去重：覆盖 LLM 跨天重复请求的同意图提醒，即使 schedule
            # 类型不同。它比时间窗口更严格：完整消息文本按字节相等，误报率约为 0。
            for j in store.jobs:
                if not j.enabled:
                    continue
                if j.payload.channel != channel or j.payload.to != to:
                    continue
                if j.payload.message != message:
                    continue
                if j.state.next_run_at_ms is None or j.state.next_run_at_ms <= now:
                    continue
                logger.info(
                    "Cron: skipped duplicate add — existing job '{}' "
                    "({}) has identical message (same channel/to, "
                    "kinds={}/{})",
                    j.name,
                    j.id,
                    j.schedule.kind,
                    schedule.kind,
                )
                self._arm_timer()
                return j

            # 跨类型时间窗口去重：覆盖护理场景中“同一意图同时 expr + at”的重复添加。
            # 窗口设为较宽的 15 分钟，因为间隔不足 15 分钟的两个真正独立提醒几乎
            # 总是 LLM 错误。少数合法情况（8:00 和 8:10 服用不同药物）会少触发
            # 一次；相比垃圾提醒，这是可接受的权衡。
            new_next = _compute_next_run(schedule, now)
            if new_next is not None:
                for j in store.jobs:
                    if not j.enabled:
                        continue
                    if j.payload.channel != channel or j.payload.to != to:
                        continue
                    existing_next = j.state.next_run_at_ms
                    if existing_next is None:
                        continue
                    if abs(existing_next - new_next) <= 15 * 60 * 1000:
                        logger.info(
                            "Cron: skipped duplicate add — existing job '{}' "
                            "({}) fires within 15min of new request "
                            "(same channel/to, kinds={}/{})",
                            j.name,
                            j.id,
                            j.schedule.kind,
                            schedule.kind,
                        )
                        self._arm_timer()
                        return j

            # 去重：周期 schedule、channel 和 recipient 都相同时，
            # 就地更新消息而不是创建重复任务。
            existing = self._find_duplicate_schedule(store.jobs, schedule, channel, to)
            if existing is not None:
                existing.payload.message = message
                existing.payload.deliver = deliver
                existing.name = name
                existing.updated_at_ms = now
                # 仅当现有任务已经触发或被禁用时才重算 next_run_at_ms，
                # 否则保留原定触发时间。
                if not existing.enabled or existing.state.next_run_at_ms is None:
                    existing.enabled = True
                    existing.state.next_run_at_ms = _compute_next_run(schedule, now)
                self._save_store()
                logger.info(
                    "Cron: updated existing job '{}' ({}) with new message (dedup on schedule+channel+to)",
                    existing.name,
                    existing.id,
                )
                self._arm_timer()
                return existing

            job = CronJob(
                id=str(uuid.uuid4())[:8],
                name=name,
                enabled=True,
                schedule=schedule,
                payload=CronPayload(
                    kind="agent_turn",
                    message=message,
                    deliver=deliver,
                    channel=channel,
                    to=to,
                    topic_tag=topic_tag,
                ),
                state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
                created_at_ms=now,
                updated_at_ms=now,
                delete_after_run=delete_after_run,
            )
            store.jobs.append(job)
            self._save_store()
        self._arm_timer()
        logger.info("Cron: added job '{}' ({})", name, job.id)
        return job

    @staticmethod
    def _find_duplicate_schedule(
        jobs: list[CronJob],
        schedule: CronSchedule,
        channel: str | None,
        to: str | None,
    ) -> CronJob | None:
        """Return an existing enabled job whose (schedule, channel, to)
        matches — used by add_job for dedup. ``at`` jobs (one-shot) are
        only deduped if their at_ms is identical (same instant)."""
        for j in jobs:
            if not j.enabled:
                continue
            if j.payload.channel != channel or j.payload.to != to:
                continue
            s = j.schedule
            if s.kind != schedule.kind:
                continue
            if schedule.kind == "cron" and s.expr == schedule.expr and s.tz == schedule.tz:
                return j
            if schedule.kind == "every" and s.every_ms == schedule.every_ms:
                return j
            if schedule.kind == "at" and s.at_ms == schedule.at_ms:
                return j
        return None

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        with self._locked():
            self._store = None
            store = self._load_store()
            before = len(store.jobs)
            store.jobs = [j for j in store.jobs if j.id != job_id]
            removed = len(store.jobs) < before
            if removed:
                self._save_store()
        if removed:
            self._arm_timer()
            logger.info("Cron: removed job {}", job_id)
        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable or disable a job."""
        with self._locked():
            self._store = None
            store = self._load_store()
            for job in store.jobs:
                if job.id == job_id:
                    now = self._now_ms()
                    job.updated_at_ms = now
                    if enabled:
                        next_run = _compute_next_run(job.schedule, now)
                        job.enabled = next_run is not None
                        job.state.next_run_at_ms = next_run
                        if next_run is None and job.state.last_status != "incompatible":
                            job.state.last_status = "expired"
                            job.state.last_error = "schedule has no future fire"
                    else:
                        job.enabled = False
                        job.state.next_run_at_ms = None
                    self._save_store()
                    self._arm_timer()
                    return job
        return None

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        """Manually run a job."""
        claimed_by_pid = os.getpid()
        claimed_at_ms = self._now_ms()
        with self._locked():
            self._store = None
            store = self._load_store()
            target = next((j for j in store.jobs if j.id == job_id), None)
            if target is None or (not force and not target.enabled):
                return False
            if target.state.last_status in {"expired", "incompatible"}:
                return False
            if (
                target.state.claimed_by_pid is not None
                and target.state.claimed_at_ms is not None
                and claimed_at_ms - target.state.claimed_at_ms < _CLAIM_TTL_MS
            ):
                return False
            target.state.claimed_by_pid = claimed_by_pid
            target.state.claimed_at_ms = claimed_at_ms
            self._save_store()

        try:
            await self._execute_job(target)
            return True
        finally:
            self._persist_job_result(target, claimed_by_pid, claimed_at_ms)
            self._release_claims([target.id], claimed_by_pid, claimed_at_ms)
            self._arm_timer()

    def status(self) -> dict:
        """Get service status."""
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }
