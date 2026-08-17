"""The three-shield gate as a composable pipeline (SOP §2 ⑥).

Order matters and each shield narrows the task set the next one judges:

1. **Gate-f (measurement validity).** Fail closed when either arm has a
   Provider failure, infrastructure failure, inconclusive result, or missing
   task. Recoverable infra is salvaged upstream by the rerun ladder
   (:func:`pico.evolver.orchestrator.scoring.eval_with_infra_rerun`); by the
   time run_gates sees the evals, any surviving failure is invalid evidence,
   not a low score.
2. **Gate-b (attribution).** Only credit the candidate on tasks where its
   mechanism actually fired — a patch can't get credit for a task it never
   touched. The set of fired tasks comes from an injectable source (the
   activation ledger / beacon); when none is given this shield is a no-op.
3. **Gate2 (significance).** Paired lift on the surviving tasks: navigator
   promotion (mean > vanilla) plus a separate credited-2σ label.

Keeping this a pure function over eval maps makes it bench-agnostic and unit
testable; the loop calls it once per candidate after the confirm eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pico.evolver.orchestrator.gates.paired import PairedResult, paired_lift
from pico.evolver.orchestrator.scoring import (
    EvaluationVerdict,
    MeasurementFailure,
    MeasurementStatus,
    MeasurementValidity,
    TaskEval,
    measurement_validity,
)


@dataclass
class GateResult:
    """Combined verdict of the three-shield pipeline for one candidate.

    ``infra_contaminated`` is retained for audit. Any failed or inconclusive
    measurement prevents promotion before Gate-b attribution narrows the set.
    """

    promoted: bool
    verdict: EvaluationVerdict
    paired: PairedResult | None
    eligible_tasks: list[str]
    candidate_validity: MeasurementValidity
    control_validity: MeasurementValidity
    infra_contaminated: list[str] = field(default_factory=list)
    unfired_excluded: list[str] = field(default_factory=list)


def _is_infrastructure_failure(ev: TaskEval | None, threshold: int) -> bool:
    return ev is not None and (ev.failure is MeasurementFailure.infrastructure or ev.infra_attempts >= threshold)


def run_gates(
    *,
    candidate_evals: dict[str, TaskEval],
    control_evals: dict[str, TaskEval],
    task_ids: list[str],
    expected_attempts: int,
    z_threshold: float = 2.0,
    fired_tasks: set[str] | None = None,
    infra_threshold: int = 1,
) -> GateResult:
    """Run Gate-f -> Gate-b -> Gate2 and return the combined verdict.

    ``expected_attempts`` is the K each requested task must reach.
    ``fired_tasks`` (Gate-b) restricts attribution when provided; ``None`` skips
    that shield. ``infra_threshold`` is the per-arm infra-trial count at or above
    which a task is reported as infra-contaminated. Surviving contamination
    fails the gate; it is never converted into an improvement score. Only Gate-b
    can leave nothing to measure; then the candidate is rejected and ``paired``
    is None.
    """
    infra_contaminated = [
        t
        for t in task_ids
        if _is_infrastructure_failure(candidate_evals.get(t), infra_threshold)
        or _is_infrastructure_failure(control_evals.get(t), infra_threshold)
    ]
    candidate_validity = measurement_validity(
        candidate_evals,
        task_ids,
        expected_attempts=expected_attempts,
    )
    control_validity = measurement_validity(
        control_evals,
        task_ids,
        expected_attempts=expected_attempts,
    )
    validity_statuses = {candidate_validity.status, control_validity.status}
    if MeasurementStatus.failed in validity_statuses:
        return GateResult(
            promoted=False,
            verdict=EvaluationVerdict.failed,
            paired=None,
            eligible_tasks=list(task_ids),
            candidate_validity=candidate_validity,
            control_validity=control_validity,
            infra_contaminated=infra_contaminated,
        )
    if MeasurementStatus.inconclusive in validity_statuses:
        return GateResult(
            promoted=False,
            verdict=EvaluationVerdict.inconclusive,
            paired=None,
            eligible_tasks=list(task_ids),
            candidate_validity=candidate_validity,
            control_validity=control_validity,
            infra_contaminated=infra_contaminated,
        )
    eligible = list(task_ids)
    unfired_excluded: list[str] = []
    if fired_tasks is not None:
        unfired_excluded = [t for t in eligible if t not in fired_tasks]
        eligible = [t for t in eligible if t in fired_tasks]

    if not eligible:
        return GateResult(
            promoted=False,
            verdict=EvaluationVerdict.rejected,
            paired=None,
            eligible_tasks=[],
            candidate_validity=candidate_validity,
            control_validity=control_validity,
            infra_contaminated=infra_contaminated,
            unfired_excluded=unfired_excluded,
        )

    paired = paired_lift(
        candidate_evals=candidate_evals,
        control_evals=control_evals,
        task_ids=eligible,
        expected_attempts=expected_attempts,
        z_threshold=z_threshold,
    )
    return GateResult(
        promoted=paired.promoted,
        verdict=paired.verdict,
        paired=paired,
        eligible_tasks=eligible,
        candidate_validity=candidate_validity,
        control_validity=control_validity,
        infra_contaminated=infra_contaminated,
        unfired_excluded=unfired_excluded,
    )


__all__ = ["GateResult", "run_gates"]
