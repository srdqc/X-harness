"""Offline paired reducer; never invokes Provider, Tools, or verifiers."""

from __future__ import annotations

import statistics
from collections import defaultdict
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
)

CORE_EFFICIENCY_METRICS = (
    "provider_logical_calls",
    "tool_calls_total",
    "repeated_repo_file_reads",
    "input_tokens",
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
    complete = expected == actual
    grouped: dict[tuple[str, int], dict[Arm, RunRecord]] = defaultdict(dict)
    for record in records:
        grouped[(record.task_id, record.repetition)][record.arm] = record
    pairs: list[dict[str, Any]] = []
    per_task_values: dict[str, dict[Arm, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    outcomes = {"a_pass_b_pass": 0, "a_fail_b_pass": 0, "a_pass_b_fail": 0, "a_fail_b_fail": 0}
    fairness_valid = True
    for (task_id, repetition), arms in sorted(grouped.items()):
        if set(arms) != {Arm.NO_REUSE, Arm.APPROVED_REUSE}:
            continue
        a, b = arms[Arm.NO_REUSE], arms[Arm.APPROVED_REUSE]
        pair_valid = a.fairness_digest == b.fairness_digest
        fairness_valid &= pair_valid
        a_pass = a.verifier_outcome == "pass"
        b_pass = b.verifier_outcome == "pass"
        outcomes[("a_pass_" if a_pass else "a_fail_") + ("b_pass" if b_pass else "b_fail")] += 1
        deltas: dict[str, Any] = {}
        for name in CORE_EFFICIENCY_METRICS:
            av = getattr(a.metrics, name)
            bv = getattr(b.metrics, name)
            if av.availability is Availability.AVAILABLE and bv.availability is Availability.AVAILABLE:
                delta = paired_delta(float(av.value), float(bv.value))
                deltas[name] = delta
                per_task_values[task_id][Arm.NO_REUSE][name].append(float(av.value))
                per_task_values[task_id][Arm.APPROVED_REUSE][name].append(float(bv.value))
            else:
                deltas[name] = {"absolute": None, "relative": None}
        pairs.append({"task_id": task_id, "repetition": repetition, "valid": pair_valid, "deltas": deltas})
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
    arm_records = {arm: tuple(item for item in records if item.arm is arm) for arm in Arm}
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
    safety_passed = all(record.safety_outcome == "pass" for record in records)
    diagnostic_classification = classify(
        valid=complete and fairness_valid,
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
        "fairness_valid": fairness_valid,
        "safety_passed": safety_passed,
        "safety_findings": safety_findings,
        "paired_outcomes": outcomes,
        "success_rates": rates,
        "pairs": pairs,
        "per_task_arm_median_pairs": task_pairs,
        "median_relative_deltas": medians,
        "mean_relative_deltas": means,
        "classification": classification.value,
        "diagnostic_classification": diagnostic_classification.value,
        "claim_scope": "under this frozen X-harness live campaign",
    }
    result["semantic_digest"] = canonical_digest(result)
    return to_primitive(result)


__all__ = ["CORE_EFFICIENCY_METRICS", "classify", "paired_delta", "reduce_campaign"]
