from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.records import VerificationState, VerifierResult

from .models import ToolMCPTask, ToolMCPTrack


@dataclass(frozen=True)
class SealedLiveStateVerifierV2:
    task: ToolMCPTask
    state_path: Path
    expected_data_digest: str
    verifier_code_digest: str

    @classmethod
    def capture(
        cls,
        task: ToolMCPTask,
        state_path: Path,
    ) -> SealedLiveStateVerifierV2:
        if task.track is not ToolMCPTrack.LIVE_SOLVABLE_V2:
            raise ValueError("live V2 verifier requires a live-solvable-v2 task")
        return cls(
            task=task,
            state_path=state_path,
            expected_data_digest=canonical_digest(dict(task.expected_state)),
            verifier_code_digest=live_verifier_v2_code_digest(),
        )

    def verify(
        self,
        event_path: Path,
    ) -> tuple[VerifierResult, tuple[dict[str, Any], ...]]:
        if self.expected_data_digest != canonical_digest(dict(self.task.expected_state)):
            return self._not_run("live_v2_fixture_expected_state_drift")
        if self.verifier_code_digest != live_verifier_v2_code_digest():
            return self._not_run("live_v2_fixture_verifier_code_drift")
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            events = tuple(
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, json.JSONDecodeError) as exc:
            return self._not_run(f"live_v2_fixture_verifier_error:{type(exc).__name__}")
        if not isinstance(state, dict) or not all(isinstance(event, dict) for event in events):
            return (
                VerifierResult(
                    state=VerificationState.FAILED,
                    findings=("invalid_live_v2_fixture_evidence_shape",),
                ),
                events,
            )
        findings: list[str] = []
        if not _contains_subset(state, self.task.expected_state):
            findings.append("expected_live_v2_fixture_state_missing")
        if self.task.role == "researcher":
            report = state.get("report")
            explanation = report.get("explanation") if isinstance(report, dict) else None
            if not isinstance(explanation, str) or not explanation.strip():
                findings.append("research_explanation_missing")
        capabilities = tuple(str(event.get("capability") or "") for event in events)
        if not _ordered_subsequence(self.task.required_capabilities, capabilities):
            findings.append("required_live_v2_capability_evidence_missing")
        if any(_prohibited(capability, self.task.prohibited_capabilities) for capability in capabilities):
            findings.append("prohibited_live_v2_capability_observed")
        if any(
            capability.endswith(".verify") and event.get("result", {}).get("passed") is not True
            for capability, event in zip(capabilities, events, strict=True)
        ):
            findings.append("live_v2_verification_tool_did_not_pass")
        return (
            VerifierResult(
                state=(VerificationState.FAILED if findings else VerificationState.PASSED),
                findings=tuple(findings),
                metrics={
                    "required_capability_count": len(self.task.required_capabilities),
                    "observed_event_count": len(events),
                },
            ),
            events,
        )

    @staticmethod
    def _not_run(finding: str) -> tuple[VerifierResult, tuple[dict[str, Any], ...]]:
        return (
            VerifierResult(
                state=VerificationState.NOT_RUN,
                findings=(finding,),
            ),
            (),
        )


def _contains_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_subset(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _ordered_subsequence(required: tuple[str, ...], observed: tuple[str, ...]) -> bool:
    position = 0
    for capability in observed:
        if position < len(required) and capability == required[position]:
            position += 1
    return position == len(required)


def _prohibited(capability: str, prohibited: tuple[str, ...]) -> bool:
    return capability in prohibited or any(
        item.startswith("*.") and capability.endswith(item[1:]) for item in prohibited
    )


def live_verifier_v2_code_digest() -> str:
    return canonical_digest(inspect.getsource(SealedLiveStateVerifierV2.verify))


__all__ = ["SealedLiveStateVerifierV2", "live_verifier_v2_code_digest"]
