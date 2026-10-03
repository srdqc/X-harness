"""Frozen arm-neutral P3R run-validity policy."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import InfraInvalidReason, RunValidity

VALIDITY_POLICY_VERSION = 1

_TERMINAL_PROVIDER_INFRA = {
    InfraInvalidReason.PROVIDER_TRANSPORT.value: InfraInvalidReason.PROVIDER_TRANSPORT,
    InfraInvalidReason.PROVIDER_SERVER.value: InfraInvalidReason.PROVIDER_SERVER,
    InfraInvalidReason.PROVIDER_MALFORMED_RESPONSE.value: (
        InfraInvalidReason.PROVIDER_MALFORMED_RESPONSE
    ),
}


@dataclass(frozen=True)
class RunValidityResult:
    validity: RunValidity
    reason: InfraInvalidReason | None = None
    policy_version: int = VALIDITY_POLICY_VERSION


def classify_run_validity(
    *,
    runtime_outcome: str,
    normalized_provider_failures: tuple[str, ...] = (),
    infrastructure_reason: InfraInvalidReason | None = None,
    mandatory_evidence_complete: bool = True,
) -> RunValidityResult:
    """Keep Agent/task failures valid and invalidate only durable infrastructure facts."""

    if infrastructure_reason is not None:
        return RunValidityResult(RunValidity.INFRA_INVALID, infrastructure_reason)
    if not mandatory_evidence_complete:
        return RunValidityResult(
            RunValidity.INFRA_INVALID,
            InfraInvalidReason.MANDATORY_EVIDENCE_FAILURE,
        )
    if runtime_outcome == "provider_failed" and normalized_provider_failures:
        reason = _TERMINAL_PROVIDER_INFRA.get(normalized_provider_failures[-1])
        if reason is not None:
            return RunValidityResult(RunValidity.INFRA_INVALID, reason)
    return RunValidityResult(RunValidity.VALID)


__all__ = ["VALIDITY_POLICY_VERSION", "RunValidityResult", "classify_run_validity"]
