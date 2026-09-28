"""Frozen deterministic P0 runtime baseline for future PicoBench comparisons."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from benchmarks.picobench.canonical import to_primitive
from benchmarks.picobench.harness import run
from benchmarks.picobench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.picobench.isolation import TrialIsolation
from benchmarks.picobench.protocol import TrialContext, TrialExecution
from benchmarks.picobench.records import (
    DeliveryOutcome,
    TrialStatus,
    TurnTerminalState,
    VerificationState,
    VerifierResult,
)
from benchmarks.picobench.registry import PackRegistry
from benchmarks.picobench.schema import (
    ExecutionPolicy,
    ExperimentRef,
    ExperimentSpec,
    PackDefinition,
    TaskSpec,
    VariantSpec,
)
from benchmarks.picobench.usage import RecordingProvider, UsageRecorder
from pico.agent.tools.base import Tool
from pico.config.pico import PicoConfig
from pico.config.schema import Config
from pico.providers.base import ErrorClassification, LLMProvider, LLMResponse, ToolCallRequest
from pico.spine import ChatType, Origin, Source, ToolEvent, ToolPhase, TurnEnded, TurnRequest
from pico.tracing import spans as tracing_spans

RUNTIME_BASELINE_SCHEMA = "pico.picobench.runtime-baseline.v1"
RUNTIME_BASELINE_PACK_ID = "runtime-baseline-v1"
RUNTIME_BASELINE_TASK_IDS = (
    "success-no-tool",
    "success-tool",
    "provider-failure",
    "success-delivery-dropped",
)


@dataclass(frozen=True)
class _ExpectedCase:
    runtime_state: TurnTerminalState
    terminal_event: str
    provider_attempts: int
    tool_executions: int
    delivery_state: DeliveryOutcome | None
    verifier_state: VerificationState


_EXPECTED_CASES = {
    "success-no-tool": _ExpectedCase(
        TurnTerminalState.COMPLETED,
        "TurnEnded",
        1,
        0,
        DeliveryOutcome.DELIVERED,
        VerificationState.PASSED,
    ),
    "success-tool": _ExpectedCase(
        TurnTerminalState.COMPLETED,
        "TurnEnded",
        2,
        1,
        DeliveryOutcome.DELIVERED,
        VerificationState.PASSED,
    ),
    "provider-failure": _ExpectedCase(
        TurnTerminalState.PROVIDER_FAILED,
        "TurnFailed",
        1,
        0,
        None,
        VerificationState.FAILED,
    ),
    "success-delivery-dropped": _ExpectedCase(
        TurnTerminalState.COMPLETED,
        "TurnEnded",
        1,
        0,
        DeliveryOutcome.DROPPED,
        VerificationState.PASSED,
    ),
}


class _BaselineProvider(LLMProvider):
    def __init__(self, task_id: str) -> None:
        super().__init__()
        self._task_id = task_id

    def get_default_model(self) -> str:
        return "scripted/runtime-baseline"

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        if self._task_id == "provider-failure":
            return LLMResponse(
                content="deterministic provider failure",
                finish_reason="error",
                error_classification=ErrorClassification("runtime_baseline_failure"),
                usage=_usage(),
            )
        if self._task_id == "success-tool" and not any(
            message.get("role") == "tool" for message in messages
        ):
            return LLMResponse(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id="runtime-baseline-tool-call",
                        name="runtime_baseline_probe",
                        arguments={"value": "frozen"},
                    )
                ],
                usage=_usage(),
            )
        return LLMResponse(
            content=f"runtime baseline completed: {self._task_id}",
            finish_reason="stop",
            usage=_usage(),
        )


class _BaselineTool(Tool):
    @property
    def name(self) -> str:
        return "runtime_baseline_probe"

    @property
    def description(self) -> str:
        return "Return one frozen deterministic runtime-baseline value."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def execute(self, **kwargs) -> str:
        return f"runtime-baseline-tool-result:{kwargs['value']}"


class RuntimeBaselineVerifier:
    """Verify frozen Runtime evidence without consulting the Agent's final text."""

    def verify(self, task_id: str, evidence: Mapping[str, Any]) -> VerifierResult:
        expected = _EXPECTED_CASES[task_id]
        findings: list[str] = []
        required = {
            "schema",
            "task_id",
            "conversation_id",
            "turn_id",
            "terminal_turn_id",
            "terminal_event",
            "terminal_outcome",
            "event_turn_ids",
            "provider_attempts",
            "provider_attempt_evidence",
            "tool_executions",
            "tool_execution_evidence",
            "latency_ms",
            "trace_id",
            "trace_turn_id",
            "trace_outcome",
            "delivery_outcome",
            "delivery_results",
            "delivery_trace_ids",
        }
        missing = sorted(required - evidence.keys())
        findings.extend(f"missing_evidence:{field}" for field in missing)
        if missing:
            return _verification_failed(findings)

        expected_turn_id = f"runtime-baseline-turn-{task_id}"
        expected_conversation = f"picobench-runtime-baseline:{task_id}"
        if evidence["schema"] != RUNTIME_BASELINE_SCHEMA:
            findings.append("schema_mismatch")
        if evidence["task_id"] != task_id:
            findings.append("task_id_mismatch")
        if evidence["conversation_id"] != expected_conversation:
            findings.append("conversation_id_mismatch")
        correlated_ids = {
            evidence["turn_id"],
            evidence["terminal_turn_id"],
            evidence["trace_turn_id"],
        }
        if correlated_ids != {expected_turn_id}:
            findings.append("turn_id_correlation_mismatch")
        if evidence["event_turn_ids"] != [expected_turn_id]:
            findings.append("event_turn_id_mismatch")
        if evidence["terminal_event"] != expected.terminal_event:
            findings.append("terminal_event_mismatch")
        if evidence["terminal_outcome"] != expected.runtime_state.value:
            findings.append("terminal_outcome_mismatch")
        if evidence["trace_outcome"] != expected.runtime_state.value:
            findings.append("trace_outcome_mismatch")
        trace_id = evidence["trace_id"]
        if not isinstance(trace_id, str) or not trace_id:
            findings.append("missing_trace_identity")

        provider_evidence = evidence["provider_attempt_evidence"]
        if evidence["provider_attempts"] != expected.provider_attempts:
            findings.append("provider_attempt_count_mismatch")
        if not isinstance(provider_evidence, list) or len(provider_evidence) != expected.provider_attempts:
            findings.append("provider_attempt_evidence_mismatch")
        elif not all(record.get("provider_dispatched") is True for record in provider_evidence):
            findings.append("provider_attempt_not_dispatched")
        elif expected.runtime_state is TurnTerminalState.PROVIDER_FAILED:
            if provider_evidence[0].get("succeeded") is not False:
                findings.append("provider_failure_evidence_mismatch")
        elif not all(record.get("succeeded") is True for record in provider_evidence):
            findings.append("provider_success_evidence_mismatch")

        tool_evidence = evidence["tool_execution_evidence"]
        if evidence["tool_executions"] != expected.tool_executions:
            findings.append("tool_execution_count_mismatch")
        if not isinstance(tool_evidence, list) or len(tool_evidence) != expected.tool_executions:
            findings.append("tool_execution_evidence_mismatch")
        elif expected.tool_executions:
            tool_result = tool_evidence[0]
            if (
                tool_result.get("name") != "runtime_baseline_probe"
                or tool_result.get("failed") is not False
                or tool_result.get("result_preview") != "runtime-baseline-tool-result:frozen"
            ):
                findings.append("tool_result_mismatch")
            if tool_result.get("turn_id") != expected_turn_id:
                findings.append("tool_turn_id_mismatch")

        expected_delivery = expected.delivery_state.value if expected.delivery_state is not None else None
        delivery_results = evidence["delivery_results"]
        delivery_trace_ids = evidence["delivery_trace_ids"]
        if evidence["delivery_outcome"] != expected_delivery:
            findings.append("delivery_outcome_mismatch")
        if expected_delivery is None:
            if delivery_results or delivery_trace_ids:
                findings.append("fabricated_delivery_evidence")
        else:
            if not isinstance(delivery_results, list) or len(delivery_results) != 1:
                findings.append("delivery_result_count_mismatch")
            else:
                result = delivery_results[0]
                if (
                    result.get("turn_id") != expected_turn_id
                    or result.get("conversation_id") != expected_conversation
                    or result.get("outcome") != expected_delivery
                ):
                    findings.append("delivery_correlation_mismatch")
            if delivery_trace_ids != [trace_id]:
                findings.append("delivery_trace_correlation_mismatch")

        latency = evidence["latency_ms"]
        if not isinstance(latency, int | float) or isinstance(latency, bool) or latency < 0:
            findings.append("invalid_latency")

        if findings:
            return _verification_failed(findings)
        if expected.verifier_state is VerificationState.FAILED:
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=(f"execution_not_successful:{expected.runtime_state.value}",),
                metrics={"evidence_contract_valid": True},
            )
        return VerifierResult(
            state=VerificationState.PASSED,
            metrics={"evidence_contract_valid": True},
        )


class RuntimeBaselinePack:
    def definition(self) -> PackDefinition:
        return PackDefinition(
            pack_id=RUNTIME_BASELINE_PACK_ID,
            tasks=tuple(TaskSpec(task_id=task_id) for task_id in RUNTIME_BASELINE_TASK_IDS),
            variants=(VariantSpec(variant_id="frozen", settings={"runtime_baseline": "v1"}),),
            pairs=(),
            identity={
                "schema": RUNTIME_BASELINE_SCHEMA,
                "task_ids": list(RUNTIME_BASELINE_TASK_IDS),
                "network_required": False,
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        task_id = context.task.task_id
        expected = _EXPECTED_CASES[task_id]
        task_index = RUNTIME_BASELINE_TASK_IDS.index(task_id)
        isolation = TrialIsolation.create(
            context.experiment.output_root / ".r",
            f"t{task_index}r{context.key.repetition}b{context.block_attempt}",
        )
        isolation.prepare()
        recorder = UsageRecorder()
        provider = RecordingProvider(_BaselineProvider(task_id), recorder=recorder)
        config, pico_config = _runtime_config(isolation.workspace, provider.get_default_model())
        outlet = RecordingOutlet(
            "picobench-runtime-baseline",
            fail=task_id == "success-delivery-dropped",
        )
        turn_id = f"runtime-baseline-turn-{task_id}"
        environment = {
            **isolation.child_environment(),
            "PICO_TRACING": "1",
            "PICO_TRACING_DIR": str(isolation.trace_root),
        }
        previous_store = tracing_spans._store
        started = time.perf_counter()
        observation = None
        try:
            with patch.dict("os.environ", environment, clear=False):
                tracing_spans._store = None
                host = await RuntimeTrialHost.build(
                    config=config,
                    pico_config=pico_config,
                    provider=provider,
                    cron_service=None,
                    outlet=outlet,
                    delivery_retries=0,
                    turn_id_factory=lambda: turn_id,
                )
                host.assembly.agent_loop.tools.register(_BaselineTool())
                try:
                    observation = await host.run(_turn_request(task_id))
                finally:
                    await host.close()
        finally:
            tracing_spans._store = previous_store
        latency_ms = (time.perf_counter() - started) * 1_000
        if observation is None:
            raise RuntimeError("runtime baseline produced no observation")

        evidence = _evidence(
            task_id=task_id,
            observation=observation,
            provider_records=recorder.records(),
            trace_root=isolation.trace_root,
            latency_ms=latency_ms,
        )
        verification = RuntimeBaselineVerifier().verify(task_id, evidence)
        if expected.runtime_state is TurnTerminalState.PROVIDER_FAILED:
            status = TrialStatus.PROVIDER_FAILURE
        elif verification.state is VerificationState.PASSED:
            status = TrialStatus.PASSED
        else:
            status = TrialStatus.TASK_FAILED
        return TrialExecution(
            status=status,
            runtime_state=observation.runtime_state,
            delivery_state=observation.delivery_state,
            verification=verification,
            observed_variant_settings=dict(context.variant.settings),
            metrics=evidence,
            artifact_refs=(isolation.root.relative_to(context.experiment.output_root).as_posix(),),
        )


def runtime_baseline_spec(output_root: Path) -> ExperimentSpec:
    return ExperimentSpec(
        suite="runtime-baseline-v1",
        repetitions=1,
        pack_ids=(RUNTIME_BASELINE_PACK_ID,),
        output_root=Path(output_root),
        identity={
            "baseline_schema": RUNTIME_BASELINE_SCHEMA,
            "provider": "scripted/runtime-baseline",
            "network_required": False,
        },
        execution=ExecutionPolicy(
            timeout_seconds=30.0,
            provider_call_max_attempts=1,
            max_comparison_block_attempts=1,
            max_provider_calls_per_trial=2,
        ),
    )


async def run_runtime_baseline(output_root: Path) -> ExperimentRef:
    registry = PackRegistry()
    registry.register(RuntimeBaselinePack())
    return await run(runtime_baseline_spec(output_root), registry=registry)


def _runtime_config(workspace: Path, model: str) -> tuple[Config, PicoConfig]:
    config = Config(
        agents={
            "defaults": {
                "workspace": str(workspace),
                "model": model,
                "max_tool_iterations": 3,
            }
        }
    )
    pico_config = PicoConfig(
        memory={"backend": None},
        plugins={"disabled": []},
        skill_forge={"router": {"enabled": False}},
    )
    pico_config.runtime.checkpoint.policy = "never"
    return config, pico_config


def _turn_request(task_id: str) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="picobench-runtime-baseline",
            chat_id=task_id,
            sender_id="picobench",
            chat_type=ChatType.DM,
        ),
        text=f"Execute frozen runtime baseline case {task_id}.",
        conversation=f"picobench-runtime-baseline:{task_id}",
    )


def _evidence(*, task_id: str, observation, provider_records, trace_root: Path, latency_ms: float) -> dict[str, Any]:
    rows = _trace_rows(trace_root)
    spine_rows = [
        row
        for row in rows
        if row.get("name") == "spine.turn"
        and row.get("attributes", {}).get("spine.turn_id") == observation.turn_id
    ]
    trace_id = spine_rows[0].get("traceId") if len(spine_rows) == 1 else None
    trace_turn_id = spine_rows[0].get("attributes", {}).get("spine.turn_id") if len(spine_rows) == 1 else None
    trace_outcome = spine_rows[0].get("attributes", {}).get("spine.outcome") if len(spine_rows) == 1 else None
    delivery_rows = [
        row
        for row in rows
        if row.get("name") == "channel.deliver"
        and row.get("attributes", {}).get("spine.turn_id") == observation.turn_id
    ]
    completed_tools = [
        event
        for event in observation.events
        if isinstance(event, ToolEvent) and event.phase is ToolPhase.COMPLETE
    ]
    terminal = observation.terminal_event
    terminal_latency = terminal.latency_ms if isinstance(terminal, TurnEnded) else None
    return {
        "schema": RUNTIME_BASELINE_SCHEMA,
        "task_id": task_id,
        "conversation_id": observation.conversation_id,
        "turn_id": observation.turn_id,
        "terminal_turn_id": terminal.turn_id,
        "terminal_event": type(terminal).__name__,
        "terminal_outcome": observation.runtime_state.value,
        "event_turn_ids": sorted(
            {
                event.turn_id
                for event in observation.events
                if getattr(event, "turn_id", None) is not None
            }
        ),
        "provider_attempts": len(provider_records),
        "provider_attempt_evidence": [_provider_evidence(record) for record in provider_records],
        "tool_executions": len(completed_tools),
        "tool_execution_evidence": [
            {
                "tool_call_id": event.tool_call_id,
                "name": event.name,
                "failed": event.failed,
                "result_preview": event.result_preview,
                "turn_id": event.turn_id,
            }
            for event in completed_tools
        ],
        "latency_ms": latency_ms,
        "terminal_latency_ms": terminal_latency,
        "trace_id": trace_id,
        "trace_turn_id": trace_turn_id,
        "trace_outcome": trace_outcome,
        "delivery_outcome": observation.delivery_state.value if observation.delivery_state is not None else None,
        "delivery_results": [to_primitive(result) for result in observation.delivery_results],
        "delivery_trace_ids": [row.get("traceId") for row in delivery_rows],
    }


def _provider_evidence(record) -> dict[str, Any]:
    return {
        "model": record.model,
        "requested_model": record.requested_model,
        "provider_dispatched": record.provider_dispatched,
        "succeeded": record.succeeded,
        "error_category": record.error_category,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
    }


def _trace_rows(trace_root: Path) -> list[dict[str, Any]]:
    path = trace_root / "logs" / "audit-spans.log"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verification_failed(findings: list[str]) -> VerifierResult:
    return VerifierResult(
        state=VerificationState.FAILED,
        findings=tuple(findings),
        metrics={"evidence_contract_valid": False},
    )


def _usage() -> dict[str, int]:
    return {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }


__all__ = [
    "RUNTIME_BASELINE_PACK_ID",
    "RUNTIME_BASELINE_SCHEMA",
    "RUNTIME_BASELINE_TASK_IDS",
    "RuntimeBaselinePack",
    "RuntimeBaselineVerifier",
    "run_runtime_baseline",
    "runtime_baseline_spec",
]
