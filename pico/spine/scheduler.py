"""Lane and worker: the per-conversation serial+cancel execution domain.

A lane runs one turn at a time and is the unit of cancellation. The worker owns
the lifecycle events and is the sole resolver of a turn's terminal future. The
pool and event sink are placeholders filled in by later sub-steps.
"""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import get_args

from loguru import logger

from pico.spine._barrier import finish_barrier
from pico.spine.events import RunnerEvent, TurnEnded, TurnEvent, TurnFailed, TurnStarted
from pico.spine.runner import Emit, TurnOutcome, TurnRunner
from pico.spine.turn import BusyPolicy, Origin, TurnRequest
from pico.tracing import semconv, trace

EventSink = Callable[[TurnEvent], Awaitable[None]]

# 使用 tuple 而不是 union，避免类型检查器把下方守卫标成不可达并诱使删除；
# 它是 lifecycle emit 的唯一强制检查。
_RUNNER_EVENT_TYPES = get_args(RunnerEvent)

# Runtime 来源共享 system pool。此处 SUBAGENT 指结果重注入 Turn；subagent
# 自身执行位于 scheduler 之外，并受独立门禁控制。
_SYSTEM_ORIGINS = (Origin.CRON, Origin.SUBAGENT)

_DEFAULT_IDLE_TTL = 300.0  # lane 空闲多久后由 reaper 回收（秒）
_SWEEP_INTERVAL = 60.0  # reaper 两次扫描的间隔（秒）
_DEPTH_WARN_THRESHOLD = 50  # lane 待处理队列达到该深度时警告一次


class SchedulerDrainingError(Exception):
    """Raised by submit once shutdown has sealed the scheduler — new turns are
    not accepted while draining.
    """


class OriginPools:
    """Per-origin concurrency gates: a USER pool and a system pool for Runtime
    origins, sized independently. No global cap (total concurrency is their sum)
    and no borrowing between them, so a user turn never waits on an LLM slot
    behind a Runtime task.
    """

    def __init__(self, user: int, system: int):
        self._user = asyncio.Semaphore(user)
        self._system = asyncio.Semaphore(system)

    def for_origin(self, origin: Origin) -> asyncio.Semaphore:
        if origin is Origin.USER:
            return self._user
        if origin in _SYSTEM_ORIGINS:
            return self._system
        # 明确失败：新 origin 必须主动选择 pool，不得通过回退静默进入 system pool。
        raise ValueError(f"no pool mapping for origin {origin!r}")


class Lane:
    def __init__(self, runner: TurnRunner, pools: OriginPools, sink: EventSink, conversation_id: str):
        self._runner = runner
        self._pools = pools
        self._sink = sink
        self._conversation_id = conversation_id
        self._pending: deque[tuple[TurnRequest, asyncio.Future]] = deque()
        self._worker: asyncio.Task | None = None
        self._run_task: asyncio.Task | None = None
        self._running_fut: asyncio.Future | None = None
        self._idle_since: float | None = None  # worker drain 后设置，供 reaper 计时
        # Turn 运行期间提交的 inject 在此等待，在工具循环间隙 drain（合并）。
        # drain 后，inject 会在 _run_turn 的 Turn 局部状态中链接到运行中 Turn，
        # 不再保存在这里。
        self._inject_mailbox: deque[tuple[TurnRequest, asyncio.Future]] = deque()

    def submit(self, req: TurnRequest, policy: BusyPolicy = BusyPolicy.APPEND) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._idle_since = None  # 再次活跃，重置 reaper 的静默计时
        running = self._run_task is not None and not self._run_task.done()
        if policy is BusyPolicy.INTERRUPT and running:
            # 抢占：取消运行中的 Turn，但只取消其 task；worker 仍是其 future 的
            # 唯一 resolver。随后把 interrupter 插到队首，置于 APPEND 积压之前。
            self._run_task.cancel()
            self._enqueue(req, fut, front=True)
        elif policy is BusyPolicy.INJECT and running:
            # 该 inject 属于运行中 Turn，暂存到 runner 在工具循环间隙 drain。
            # 若 Turn 结束时仍未 drain，worker 会将其回退为 APPEND Turn。
            # 运行中 Turn 的 worker 仍然存活，无需重新启动。
            self._inject_mailbox.append((req, fut))
            return fut
        else:
            # APPEND，或空闲 lane 上无目标 Turn 的 INTERRUPT/INJECT：
            # 像普通 Turn 一样入队运行。
            self._enqueue(req, fut)
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run_worker())
        return fut

    def _enqueue(self, req: TurnRequest, fut: asyncio.Future, *, front: bool = False) -> None:
        # 所有 _pending 增长的唯一入口：APPEND 放尾部、INTERRUPT 放头部，以及
        # worker 的 inject 回退。这样深度检查能看到每次 +1，并在每次向上越过阈值
        # 时恰好警告一次，无需额外标志。所有 _pending 增长都必须经过这里，
        # 否则可能跳过 == 检查。
        if front:
            self._pending.appendleft((req, fut))
        else:
            self._pending.append((req, fut))
        if len(self._pending) == _DEPTH_WARN_THRESHOLD:
            logger.warning(
                "lane {} pending queue reached depth {}",
                self._conversation_id,
                _DEPTH_WARN_THRESHOLD,
            )

    def cancel_turn(self, fut: asyncio.Future) -> None:
        """Cancel one turn (its handle's): drop it if queued, cancel its task if
        running. Idempotent on an already-resolved future. Never resolves the
        running turn's future — the worker stays its sole resolver.
        """
        if fut.done():
            return
        for i, (_req, queued) in enumerate(self._pending):
            if queued is fut:
                del self._pending[i]
                fut.set_result(None)
                return
        for i, (_req, pending_inject) in enumerate(self._inject_mailbox):
            if pending_inject is fut:
                # inject 仍在 mailbox、尚未合并时可以取消自身。drain/合并后它不再
                # 位于此处，此时取消其 handle 不产生效果，也无法杀死 Host Turn。
                del self._inject_mailbox[i]
                fut.set_result(None)
                return
        if fut is self._running_fut and self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()

    def cancel_running(self) -> int:
        """Cancel the running turn's task — never its future (the worker stays the
        sole resolver) — returning 1 if there was one to cancel, else 0.
        """
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            return 1
        return 0

    def running_future(self) -> asyncio.Future | None:
        """The in-flight turn's terminal future, or None if the lane is idle.
        Shutdown awaits these within its grace window.
        """
        return self._running_fut

    def has_pending_or_running(self) -> bool:
        """True while this Lane owns work that has not reached a terminal."""
        return self._running_fut is not None or bool(self._pending) or bool(self._inject_mailbox)

    def drain_pending(self) -> int:
        """Resolve every not-yet-running turn (queued + mailboxed injects) as
        cancelled and return the count, leaving the running turn untouched.

        Already-drained injects are chained to the running turn and follow its
        fate inside _run_turn — not here, so each future resolves exactly once.
        """
        stopped = 0
        while self._pending:
            _req, fut = self._pending.popleft()
            if not fut.done():
                fut.set_result(None)  # 入队 Turn 从未运行，以 cancelled 完成
            stopped += 1
        while self._inject_mailbox:
            _req, fut = self._inject_mailbox.popleft()
            if not fut.done():
                fut.set_result(None)  # 未 drain 的 inject，以 cancelled 完成且不再恢复
            stopped += 1
        return stopped

    def cancel(self) -> int:
        """/stop: cancel the running turn and drain the queue, resolving every
        waiting future as cancelled; return how many turns were stopped.
        """
        return self.cancel_running() + self.drain_pending()

    def idle_for(self, now: float) -> float | None:
        """Seconds since the worker drained and the lane went idle, or None while
        it is still active. The reaper reads this to reclaim long-silent lanes.
        """
        if self._idle_since is None:
            return None
        return now - self._idle_since

    async def _run_worker(self) -> None:
        while self._pending:
            req, fut = self._pending.popleft()
            self._running_fut = fut
            # 同步创建 task（此前不 await），使 Turn 一离开队列便可通过
            # _run_task 观察和执行取消。
            self._run_task = asyncio.create_task(self._run_turn(req))
            outcome: TurnOutcome | None = None
            try:
                outcome = await self._run_task  # 取消/失败为 None，成功时为 outcome
            except asyncio.CancelledError:
                if not self._run_task.cancelled():
                    # worker 自身因进程关闭而被取消：把取消级联到 payload 并等待清理。
                    # payload 的 finally 会发出 TurnFailed 并完成所有链式 inject，
                    # 避免留下僵尸任务，然后重新抛出。正常关闭只取消 payload 而非
                    # worker；该路径属于一次性强制终止。
                    self._run_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._run_task
                    raise
                # payload 被取消：Turn 已自行发出终态。
            finally:
                try:
                    # 作为唯一 resolver，在每个退出路径完成 future，包括 task body
                    # 及其 handler 运行前就发生的取消。合并后的 inject 在 _run_turn
                    # 内完成，不在这里处理。
                    if not fut.done():
                        fut.set_result(outcome)
                    # Turn 未 drain 的 inject 回退为新的 APPEND Turn，确保消息不丢失。
                    # USER inject 会记录该回退，便于排查“为何没有完成注入”。
                    while self._inject_mailbox:
                        inject_req, inject_fut = self._inject_mailbox.popleft()
                        if inject_req.origin is Origin.USER:
                            logger.info(
                                "inject fell back to append (not merged): origin={}",
                                inject_req.origin,
                            )
                        self._enqueue(inject_req, inject_fut)
                finally:
                    self._run_task = None
                    self._running_fut = None
        # 空闲退出时记录 reaper 的静默起点。队列检查与 return 之间不得 await，
        # 避免与退出竞争的 submit 丢失唤醒；同样的无 await 区间也保证该时间戳
        # 与 submit 清除动作之间的原子性。
        self._idle_since = time.monotonic()

    def _make_emit(self, req: TurnRequest) -> Emit:
        async def emit(event: RunnerEvent) -> None:
            if not isinstance(event, _RUNNER_EVENT_TYPES):
                raise TypeError(f"a runner may not emit lifecycle events; got {type(event).__name__}")
            # 补全 runner 未设置的身份字段，只填 None，绝不覆盖：source 是 Turn
            # 回复地址，conversation_id 是其 lane key，也是 hub 关联 stream 的依据。
            if event.source is None:
                event = replace(event, source=req.source)
            if event.conversation_id is None:
                event = replace(event, conversation_id=self._conversation_id)
            await self._sink(event)

        return emit

    async def _emit_terminal(self, event: TurnFailed | TurnEnded) -> None:
        try:
            await self._sink(event)
        except Exception:
            logger.exception("terminal event sink failed: event={}", type(event).__name__)

    async def _run_turn(self, req: TurnRequest) -> TurnOutcome | None:
        chained: list[asyncio.Future] = []

        def drain() -> list[TurnRequest]:
            # 读取并移除待处理 inject，把每个都链接到当前 Turn。从 mailbox 移除后，
            # /stop 不会再同时完成它们，从而保证每个 future 只有一个 resolver。
            drained = list(self._inject_mailbox)
            self._inject_mailbox.clear()
            chained.extend(fut for _req, fut in drained)
            return [req for req, _fut in drained]

        outcome: TurnOutcome | None = None
        started = False
        # Turn 根 span 位于这里而非 Agent Loop，因为 lifecycle event 及终态都在
        # 此处发出。Loop 的 session.turn 在 runner 内打开并成为其子节点。
        # root=True 是因为 submit() 会在提交方上下文中启动 lane worker；若从活动
        # span 内提交 Turn（如 subagent 重注入结果），否则会错误加入调用方 trace。
        # 每个 Turn 必须恰好拥有一个独立 trace。
        with trace.span(
            "spine.turn",
            semconv.spine_turn_open(req, self._conversation_id),
            kind="session",
            root=True,
            session_key=self._conversation_id,
            channel=req.source.channel,
            chat_id=req.source.chat_id,
        ) as turn_span:
            try:
                async with self._pools.for_origin(req.origin):
                    await self._sink(TurnStarted(conversation_id=self._conversation_id))
                    started = True
                    run_start = time.monotonic()
                    outcome = await self._runner.run(req, self._make_emit(req), drain)
            except asyncio.CancelledError:
                turn_span.set(semconv.spine_turn_cancelled(started=started))
                if started:  # 只与 TurnStarted 配对；启动前取消不发出事件
                    await self._emit_terminal(
                        TurnFailed(error="cancelled", cancelled=True, conversation_id=self._conversation_id)
                    )
                raise
            except Exception as exc:
                turn_span.set(semconv.spine_turn_failed(exc, started=started))
                turn_span.error(exc)
                if started:
                    await self._emit_terminal(
                        TurnFailed(error=str(exc), cancelled=False, conversation_id=self._conversation_id)
                    )
                return None
            finally:
                # 已 drain 的 inject 共享当前 Turn 的 outcome，取消/失败时为 None；
                # 由合并它的 Turn 在此完成，而不是由 worker 完成。
                for inject_fut in chained:
                    if not inject_fut.done():
                        inject_fut.set_result(outcome)
            latency_ms = (time.monotonic() - run_start) * 1000
            turn_span.set(semconv.spine_turn_ended(outcome, latency_ms))
            await self._emit_terminal(
                TurnEnded(
                    usage=outcome.usage,
                    latency_ms=latency_ms,
                    explicit_reply=outcome.explicit_reply,
                    conversation_id=self._conversation_id,
                    tool_calls=outcome.tool_calls,
                    tool_failures=outcome.tool_failures,
                )
            )
            return outcome


class TurnHandle:
    """Returned by submit; the caller's view of one turn."""

    def __init__(self, lane: Lane, fut: asyncio.Future):
        self._lane = lane
        self._fut = fut

    async def result(self) -> TurnOutcome | None:
        return await asyncio.shield(self._fut)

    def cancel(self) -> None:
        self._lane.cancel_turn(self._fut)


class Scheduler:
    """The single entry: submit a request, get a handle. Routes each request to
    its conversation lane (created on demand), which serialises and gates it.
    """

    def __init__(self, runner: TurnRunner, pools: OriginPools, sink: EventSink):
        self._loop = asyncio.get_running_loop()  # home loop；submit 必须来自这里
        self._runner = runner
        self._pools = pools
        self._sink = sink
        self._lanes: dict[str, Lane] = {}
        self._draining = False
        self._reaper: asyncio.Task | None = None
        self._shutdown_task: asyncio.Task[None] | None = None

    def submit(self, req: TurnRequest) -> TurnHandle:
        if asyncio.get_running_loop() is not self._loop:
            # 从其他 loop 调用（如 channel 的 ws 线程运行自己的 loop）会在错误
            # loop 上构建 lane；应明确失败，不做静默桥接。
            raise RuntimeError("submit must be called from the scheduler's event loop")
        if self._draining:
            logger.info("submit rejected: scheduler draining (origin={})", req.origin)
            raise SchedulerDrainingError("scheduler is draining; new turns are not accepted")
        policy = self._effective_busy(req)
        conversation_id = self._conversation_id(req)
        lane = self._lanes.get(conversation_id)
        if lane is None:
            lane = Lane(self._runner, self._pools, self._sink, conversation_id)
            self._lanes[conversation_id] = lane
        handle = TurnHandle(lane, lane.submit(req, policy))
        # 与第一个 lane 一起惰性启动 reaper，模式与 lane worker 相同；
        # 没有剩余 lane 时它会自行终止。
        if self._reaper is None or self._reaper.done():
            self._reaper = self._loop.create_task(self._reap_loop())
        return handle

    def cancel_conversation(self, conversation_id: str) -> int:
        """/stop: cancel the running turn and drain the queue for a conversation's
        lane, returning how many turns were stopped (0 if no such lane exists).
        The spine-native equivalent of the bus drainer's per-session _handle_stop.
        """
        lane = self._lanes.get(conversation_id)
        return lane.cancel() if lane is not None else 0

    def has_inflight(self, conversation_id: str) -> bool:
        """True if a turn is currently running for this conversation's lane.
        The inbound gate uses this to submit a mid-turn message as
        BusyPolicy.INJECT instead of queuing a fresh turn.
        """
        lane = self._lanes.get(conversation_id)
        return lane is not None and lane.running_future() is not None

    def has_pending_or_running(self, conversation_id: str) -> bool:
        """True if the conversation's Lane owns queued or running work."""
        lane = self._lanes.get(conversation_id)
        return lane is not None and lane.has_pending_or_running()

    async def _reap_loop(self) -> None:
        # 与 lane worker 一样自行终止：有 lane 可回收时运行，无 lane 时退出；
        # 下一次 submit 会重新启动。
        while self._lanes:
            await asyncio.sleep(_SWEEP_INTERVAL)
            self._sweep(time.monotonic())

    async def _finish_shutdown(self, grace: float) -> None:
        for lane in self._lanes.values():
            lane.drain_pending()  # 阶段 2：清除队列和 mailbox 中的工作
        # seal 与 drain 之间不得 await，使未启动工作在宽限窗口允许运行中 Turn
        # 结束并把 mailbox 回退成新 Turn 前，以原子方式完成。
        if self._reaper is not None and not self._reaper.done():
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper  # 等待取消完成，确保 shutdown 后任务真正结束
        running = [f for lane in self._lanes.values() if (f := lane.running_future()) is not None]
        if running:  # 阶段 3：为运行中 Turn 提供宽限窗口
            await asyncio.wait(running, timeout=grace)
        for lane in self._lanes.values():  # 阶段 4：级联取消仍未结束的任务
            lane.cancel_running()
        survivors = [fut for fut in running if not fut.done()]
        if survivors:
            await asyncio.wait(survivors)

    async def shutdown(self, grace: float) -> None:
        """Seal, drain, and cascade before propagating caller cancellation.

        Every turn's future resolves on one of the four exit paths, so result()
        never hangs. Concurrent callers share the first shutdown's grace window.
        """
        if self._shutdown_task is None:
            self._draining = True  # 阶段 1：seal，此后 submit 快速失败
            self._shutdown_task = asyncio.create_task(self._finish_shutdown(grace))
        await finish_barrier(self._shutdown_task)

    def _effective_busy(self, req: TurnRequest) -> BusyPolicy:
        """Resolve the busy policy actually applied. INJECT and INTERRUPT are
        USER-only — a system origin must not interrupt or inject into a user's
        turn — so a non-USER request asking for either is demoted to APPEND. The
        demotion is logged with the requested and applied policy so "I asked for
        INTERRUPT, why nothing happened" is debuggable.
        """
        if req.busy is BusyPolicy.APPEND or req.origin is Origin.USER:
            return req.busy
        logger.info(
            "busy policy demoted: origin={} requested={} applied={}",
            req.origin,
            req.busy,
            BusyPolicy.APPEND,
        )
        return BusyPolicy.APPEND

    def _conversation_id(self, req: TurnRequest) -> str:
        if req.conversation is not None:
            return req.conversation
        # scheduler 与 channel 无关。以 thread 或 topic（chat 内子会话）为键的
        # channel 需自行格式化该键，并通过上方显式 conversation 传入。这里仅生成
        # 中性的默认值，不引入任何 channel 专属知识。
        return f"{req.source.channel}:{req.source.chat_id}"

    def _sweep(self, now: float) -> int:
        """Reap lanes idle (worker drained and gone) past _DEFAULT_IDLE_TTL; return
        how many were dropped. Synchronous and await-free, so it cannot interleave
        with the equally synchronous submit: a request can never vanish into a lane
        being reaped — submit runs either wholly before (lane active, skipped) or
        wholly after (a fresh lane is built). The atomicity is structural, not locked.
        """
        reaped = 0
        for conversation_id, lane in list(self._lanes.items()):
            idle = lane.idle_for(now)
            if idle is not None and idle >= _DEFAULT_IDLE_TTL:
                del self._lanes[conversation_id]
                reaped += 1
        return reaped
