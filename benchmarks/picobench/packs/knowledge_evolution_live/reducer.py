"""Offline paired reducer; never invokes Provider, Tools, or verifiers."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from benchmarks.picobench.canonical import canonical_digest, to_primitive

from .protocol import (
    BENEFICIAL_MEDIAN_IMPROVEMENT,
    MATERIAL_MEDIAN_REGRESSION,
    REPEATED_BASELINE_PASS_TREATMENT_FAIL,
)
from .schema import (
    Arm,
    Availability,
    BenefitClassification,
    CampaignManifest,
    CampaignMode,
    RunRecord,
    RunValidity,
)

CORE_EFFICIENCY_METRICS = (
    "provider_logical_calls",
    "tool_calls_total",
    "repeated_repo_file_reads",
    "input_tokens",
    "turn_latency_ms",
)
REPLICATION_METRICS = (
    "provider_logical_calls",
    "tool_calls_total",
    "agent_iterations",
    "first_edit_iteration",
    "input_tokens",
    "output_tokens",
    "total_repo_file_reads",
    "unique_repo_files_read",
    "repeated_repo_file_reads",
    "skill_candidates_retrieved",
    "skills_referenced",
    "skills_activated",
    "p3_approximate_context_tokens",
    "turn_latency_ms",
)


def paired_delta(control: float, treatment: float) -> dict[str, float | None]:
    return {
        "absolute": treatment - control,
        "relative": ((treatment - control) / control if control != 0 else None),
    }


def classify(
    *,
    valid: bool,
    control_success_rate: float,
    treatment_success_rate: float,
    baseline_pass_treatment_fail: int,
    median_relative_deltas: dict[str, float | None],
    safety_passed: bool,
) -> BenefitClassification:
    if not valid:
        return BenefitClassification.INVALID
    if not safety_passed:
        return BenefitClassification.REGRESSIVE
    if (
        treatment_success_rate < control_success_rate
        or baseline_pass_treatment_fail >= REPEATED_BASELINE_PASS_TREATMENT_FAIL
    ):
        return BenefitClassification.REGRESSIVE
    available = {key: value for key, value in median_relative_deltas.items() if value is not None}
    severe = sum(value > MATERIAL_MEDIAN_REGRESSION for value in available.values())
    if severe >= 2:
        return BenefitClassification.REGRESSIVE
    improved = any(value <= BENEFICIAL_MEDIAN_IMPROVEMENT for value in available.values())
    unexplained_regression = any(
        value > MATERIAL_MEDIAN_REGRESSION for value in available.values()
    )
    return (
        BenefitClassification.BENEFICIAL
        if improved and not unexplained_regression
        else BenefitClassification.NEUTRAL
    )


def reduce_campaign(manifest: CampaignManifest, records: tuple[RunRecord, ...]) -> dict[str, Any]:
    manifest.validate()
    for record in records:
        record.validate()
    expected = {item.run_id for item in manifest.planned_runs}
    actual = {item.run_id for item in records}
    complete = expected <= actual
    grouped: dict[tuple[str, int], dict[Arm, list[RunRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[(record.task_id, record.repetition)][record.arm].append(record)
    pairs: list[dict[str, Any]] = []
    per_task_values: dict[str, dict[Arm, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    outcomes = {"a_pass_b_pass": 0, "a_fail_b_pass": 0, "a_pass_b_fail": 0, "a_fail_b_fail": 0}
    fairness_valid = True
    incomplete_pairs: list[dict[str, Any]] = []
    selected_records: list[RunRecord] = []
    for (task_id, repetition), arms in sorted(grouped.items()):
        if set(arms) != {Arm.NO_REUSE, Arm.APPROVED_REUSE}:
            incomplete_pairs.append({"task_id": task_id, "repetition": repetition})
            continue
        valid_arms = {
            arm: tuple(item for item in values if item.run_validity is RunValidity.VALID)
            for arm, values in arms.items()
        }
        if any(not values for values in valid_arms.values()):
            incomplete_pairs.append({"task_id": task_id, "repetition": repetition})
            continue
        a = sorted(
            valid_arms[Arm.NO_REUSE], key=lambda item: item.replacement_for_run_id is not None
        )[-1]
        b = sorted(
            valid_arms[Arm.APPROVED_REUSE], key=lambda item: item.replacement_for_run_id is not None
        )[-1]
        selected_records.extend((a, b))
        pair_valid = a.fairness_digest == b.fairness_digest
        fairness_valid &= pair_valid
        a_pass = a.verifier_outcome == "pass"
        b_pass = b.verifier_outcome == "pass"
        outcomes[("a_pass_" if a_pass else "a_fail_") + ("b_pass" if b_pass else "b_fail")] += 1
        deltas: dict[str, Any] = {}
        for name in REPLICATION_METRICS:
            av = getattr(a.metrics, name)
            bv = getattr(b.metrics, name)
            if av.availability is Availability.AVAILABLE and bv.availability is Availability.AVAILABLE:
                delta = paired_delta(float(av.value), float(bv.value))
                deltas[name] = delta
                if name in CORE_EFFICIENCY_METRICS:
                    per_task_values[task_id][Arm.NO_REUSE][name].append(float(av.value))
                    per_task_values[task_id][Arm.APPROVED_REUSE][name].append(float(bv.value))
            else:
                deltas[name] = {"absolute": None, "relative": None}
        treatment_only_paths = sorted(set(b.repository_read_paths) - set(a.repository_read_paths))
        pairs.append(
            {
                "task_id": task_id,
                "repetition": repetition,
                "valid": pair_valid,
                "verified_success": {
                    Arm.NO_REUSE.value: a_pass,
                    Arm.APPROVED_REUSE.value: b_pass,
                },
                "deltas": deltas,
                "arms": {
                    Arm.NO_REUSE.value: _replication_evidence(a),
                    Arm.APPROVED_REUSE.value: _replication_evidence(b),
                },
                "treatment_only_explored_paths": treatment_only_paths,
                "treatment_only_explored_path_count": len(treatment_only_paths),
            }
        )
    expected_pair_keys = {
        (item.task_id, item.repetition) for item in manifest.planned_runs
    }
    incomplete_keys = {(item["task_id"], item["repetition"]) for item in incomplete_pairs}
    completed_pair_keys = {(item["task_id"], item["repetition"]) for item in pairs}
    for task_id, repetition in sorted(expected_pair_keys - incomplete_keys - completed_pair_keys):
        incomplete_pairs.append({"task_id": task_id, "repetition": repetition})
    task_pairs: list[dict[str, Any]] = []
    median_inputs: dict[str, list[float]] = defaultdict(list)
    for task_id, arms in sorted(per_task_values.items()):
        deltas: dict[str, Any] = {}
        for name in CORE_EFFICIENCY_METRICS:
            control = arms[Arm.NO_REUSE].get(name, [])
            treatment = arms[Arm.APPROVED_REUSE].get(name, [])
            if control and treatment:
                delta = paired_delta(statistics.median(control), statistics.median(treatment))
                deltas[name] = delta
                if delta["relative"] is not None:
                    median_inputs[name].append(float(delta["relative"]))
            else:
                deltas[name] = {"absolute": None, "relative": None}
        task_pairs.append({"task_id": task_id, "arm_median_deltas": deltas})
    arm_records = {
        arm: tuple(item for item in selected_records if item.arm is arm) for arm in Arm
    }
    rates = {
        arm.value: (
            sum(item.verifier_outcome == "pass" for item in values) / len(values) if values else 0.0
        )
        for arm, values in arm_records.items()
    }
    medians = {name: (statistics.median(values) if values else None) for name, values in median_inputs.items()}
    means = {name: (statistics.mean(values) if values else None) for name, values in median_inputs.items()}
    safety_findings = tuple(
        sorted({finding for record in records for finding in record.safety_findings})
    )
    safety_passed = all(record.safety_outcome == "pass" for record in selected_records)
    valid_complete_pairs = len(pairs)
    expected_pair_count = len(expected_pair_keys)
    efficacy_complete = complete and valid_complete_pairs == expected_pair_count
    diagnostic_classification = classify(
        valid=efficacy_complete and fairness_valid,
        control_success_rate=rates[Arm.NO_REUSE.value],
        treatment_success_rate=rates[Arm.APPROVED_REUSE.value],
        baseline_pass_treatment_fail=outcomes["a_pass_b_fail"],
        median_relative_deltas=medians,
        safety_passed=safety_passed,
    )
    classification = (
        BenefitClassification.NOT_EVALUATED
        if manifest.mode is CampaignMode.PILOT
        else diagnostic_classification
    )
    result = {
        "schema": "pico.picobench.p3r-reduction.v1",
        "campaign_id": manifest.campaign_id,
        "campaign_mode": manifest.mode.value,
        "claim_eligible": manifest.mode is not CampaignMode.PILOT,
        "campaign_label": (
            "PILOT / NOT FOR BENEFIT CLAIM"
            if manifest.mode is CampaignMode.PILOT
            else "OFFICIAL"
        ),
        "complete": complete,
        "planned_runs": len(manifest.planned_runs),
        "completed_runs": len(records),
        "valid_runs": sum(item.run_validity is RunValidity.VALID for item in records),
        "infra_invalid_runs": sum(
            item.run_validity is RunValidity.INFRA_INVALID for item in records
        ),
        "replacement_runs": sum(item.replacement_for_run_id is not None for item in records),
        "valid_complete_pairs": valid_complete_pairs,
        "incomplete_pairs": incomplete_pairs,
        "fairness_valid": fairness_valid,
        "safety_passed": safety_passed,
        "safety_findings": safety_findings,
        "paired_outcomes": outcomes,
        "success_rates": rates,
        "pairs": pairs,
        "replication_questions": manifest.replication_questions,
        "candidate_set_overlap": {
            "retrieved": _candidate_set_overlap(
                arm_records[Arm.APPROVED_REUSE], "retrieved_candidate_ids"
            ),
            "injected": _candidate_set_overlap(
                arm_records[Arm.APPROVED_REUSE], "injected_candidate_ids"
            ),
        },
        "nav_cost_replication": tuple(
            item for item in pairs if item["task_id"] == "p3r-nav-01"
        ),
        "per_task_arm_median_pairs": task_pairs,
        "median_relative_deltas": medians,
        "mean_relative_deltas": means,
        "classification": classification.value,
        "diagnostic_classification": diagnostic_classification.value,
        "claim_scope": "under this frozen X-harness live campaign",
        "infra_invalid_reason_counts": dict(
            sorted(
                Counter(
                    item.infra_invalid_reason.value
                    for item in records
                    if item.infra_invalid_reason is not None
                ).items()
            )
        ),
        "provider_reliability": {
            "logical_calls": sum(
                int(item.metrics.provider_logical_calls.value or 0) for item in records
            ),
            "attempts": sum(int(item.metrics.provider_attempts.value or 0) for item in records),
            "failed_attempts": sum(
                int(item.metrics.provider_failed_attempts.value or 0) for item in records
            ),
            "recovered_failures": sum(item.recovered_provider_failure_count for item in records),
            "terminal_failures": sum(item.terminal_provider_failure for item in records),
        },
        "token_accounting_completeness": dict(
            sorted(Counter(item.metrics.token_accounting_status.value for item in records).items())
        ),
        "repository_read_metric_availability": dict(
            sorted(
                Counter(item.metrics.repeated_repo_file_reads.availability.value for item in records).items()
            )
        ),
        "iteration_exhaustion_count": sum(item.metrics.iteration_exhausted for item in records),
        "verifier_finding_categories": dict(
            sorted(Counter(finding for item in records for finding in item.verifier_findings).items())
        ),
    }
    result["semantic_digest"] = canonical_digest(result)
    return to_primitive(result)


def _replication_evidence(record: RunRecord) -> dict[str, Any]:
    return {
        "metrics": {
            name: to_primitive(getattr(record.metrics, name)) for name in REPLICATION_METRICS
        },
        "retrieved_candidate_ids": record.retrieved_candidate_ids,
        "injected_candidate_ids": record.injected_candidate_ids,
        "referenced_skill_ids": record.referenced_candidate_ids,
        "activated_skill_ids": record.activated_candidate_ids,
        "changed_paths": record.changed_paths,
        "repository_read_paths": record.repository_read_paths,
        "iteration_exhausted": record.metrics.iteration_exhausted,
        "verifier_findings": record.verifier_findings,
    }


def _candidate_set_overlap(
    records: tuple[RunRecord, ...],
    field: str,
) -> dict[str, Any]:
    values = tuple(
        (f"{item.task_id}:r{item.repetition}", frozenset(getattr(item, field)))
        for item in sorted(records, key=lambda value: (value.task_id, value.repetition))
    )
    comparisons: list[dict[str, Any]] = []
    for index, (left_id, left) in enumerate(values):
        for right_id, right in values[index + 1 :]:
            union = left | right
            comparisons.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "jaccard": len(left & right) / len(union) if union else 1.0,
                    "identical": left == right,
                }
            )
    return {
        "comparisons": comparisons,
        "identical_pair_count": sum(item["identical"] for item in comparisons),
        "distinct_set_count": len({tuple(sorted(items)) for _, items in values}),
    }


__all__ = [
    "CORE_EFFICIENCY_METRICS",
    "REPLICATION_METRICS",
    "classify",
    "paired_delta",
    "reduce_campaign",
]
