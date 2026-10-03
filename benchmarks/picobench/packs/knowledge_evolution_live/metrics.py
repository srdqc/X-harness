"""Structured-evidence metric extraction for P3R live runs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from pico.knowledge_evolution import CandidateType, KnowledgeRecordStore
from pico.tracing import evidence, replay

from .schema import Availability, MetricValue, RunMetrics, TokenAccountingStatus


def available(value: int | float) -> MetricValue:
    return MetricValue(value, Availability.AVAILABLE)


def unavailable() -> MetricValue:
    return MetricValue(None, Availability.NOT_AVAILABLE)


def partial(value: int | float) -> MetricValue:
    return MetricValue(value, Availability.PARTIAL)


def repeated_repository_reads(paths: tuple[str, ...]) -> tuple[int, int, int]:
    normalized = tuple(Path(path.replace("\\", "/")).as_posix().casefold() for path in paths)
    counts = Counter(normalized)
    return len(counts), len(normalized), sum(max(count - 1, 0) for count in counts.values())


def extract_run_metrics(
    *,
    trace_root: Path,
    turn_id: str,
    knowledge_state_root: Path,
    repository_read_paths: tuple[str, ...] | None = None,
) -> tuple[RunMetrics, dict[str, tuple[str, ...]]]:
    """Use durable receipts only; unavailable evidence remains unavailable."""

    readback = evidence.read_turn_evidence(trace_root, turn_id)
    replayed = replay.replay_turn(trace_root, turn_id)
    attempts = readback.provider_attempts
    logical_calls = {item.logical_call_id for item in attempts if item.logical_call_id}
    attempts_with_usage = sum(item.usage_available is True for item in attempts)
    attempts_without_usage = len(attempts) - attempts_with_usage
    if attempts_with_usage == len(attempts) and attempts:
        token_status = TokenAccountingStatus.COMPLETE
    elif attempts_with_usage:
        token_status = TokenAccountingStatus.PARTIAL
    else:
        token_status = TokenAccountingStatus.NOT_AVAILABLE
    input_tokens = output_tokens = cached_tokens = 0
    cached_available = bool(attempts_with_usage)
    for attempt in attempts:
        if attempt.usage_available is True:
            input_tokens += int(attempt.usage.get("input_tokens", attempt.usage.get("prompt_tokens", 0)))
            output_tokens += int(attempt.usage.get("output_tokens", attempt.usage.get("completion_tokens", 0)))
            if "cached_tokens" not in attempt.usage and "cache_read_input_tokens" not in attempt.usage:
                cached_available = False
            cached_tokens += int(
                attempt.usage.get("cached_tokens", attempt.usage.get("cache_read_input_tokens", 0))
            )

    token_metric = {
        TokenAccountingStatus.COMPLETE: available,
        TokenAccountingStatus.PARTIAL: partial,
        TokenAccountingStatus.NOT_AVAILABLE: lambda _value: unavailable(),
    }[token_status]

    read_receipts = tuple(
        item for item in replayed.tool_executions if item.resolved_name == "read_file"
    )
    evidence_paths = tuple(
        item.repository_read_path
        for item in read_receipts
        if item.repository_read_path is not None
    )
    if repository_read_paths is None:
        # The receipt field is authoritative. A run with complete Tool receipts
        # and no read_file calls has a measured zero, not missing data.
        if replayed.evidence_status.value == "complete" and len(evidence_paths) == len(read_receipts):
            unique, total, repeated = repeated_repository_reads(evidence_paths)
            read_values = (available(unique), available(total), available(repeated))
        else:
            read_values = (unavailable(), unavailable(), unavailable())
    else:
        unique, total, repeated = repeated_repository_reads(repository_read_paths)
        read_values = (available(unique), available(total), available(repeated))

    store = KnowledgeRecordStore(knowledge_state_root)
    usages = store.list_usages(turn_id=turn_id)
    retrieval_ids = tuple(sorted({item.retrieval_id for item in usages}))
    candidate_ids = tuple(sorted({item.candidate_id for item in usages}))
    retrieved_ids: set[str] = set()
    for retrieval_id in retrieval_ids:
        receipt = store.read_retrieval(retrieval_id)
        if receipt is not None:
            retrieved_ids.update(receipt.selected_candidate_ids)
    retrieved_candidates = tuple(
        candidate
        for candidate_id in sorted(retrieved_ids)
        if (candidate := store.read_candidate(candidate_id)) is not None
    )
    skill_usages = tuple(
        item for item in usages if item.knowledge_type is CandidateType.SKILL_CANDIDATE
    )
    skill_modes = Counter(item.usage_mode.value for item in skill_usages)
    skill_counts = Counter(item.candidate_id for item in skill_usages)
    repeated_skill_reads = sum(max(count - 1, 0) for count in skill_counts.values())
    visible_candidate_ids = {
        item.candidate_id
        for item in usages
        if item.usage_mode.value in {"injected", "referenced", "activated"}
    }
    approximate_context_tokens = 0
    approximate_available = True
    for candidate_id in visible_candidate_ids:
        candidate = store.read_candidate(candidate_id)
        if candidate is None:
            approximate_available = False
            break
        approximate_context_tokens += max(
            1,
            (len(candidate.title) + len(candidate.reusable_content) + 3) // 4,
        )

    timestamps = [event.timestamp for event in readback.events if event.event_type in {evidence.TURN_STARTED, evidence.TURN_TERMINAL}]
    latency = unavailable()
    if len(timestamps) >= 2:
        start = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
        latency = available(round((end - start).total_seconds() * 1000, 3))

    exhausted = next(
        (
            item
            for item in readback.events
            if item.event_type == evidence.AGENT_ITERATION_BUDGET_EXHAUSTED
        ),
        None,
    )
    final_synthesis = bool(
        exhausted
        and any(
            item.started_sequence is not None and item.started_sequence > exhausted.sequence
            for item in attempts
        )
    )
    agent_iterations = (
        int(exhausted.metadata.get("agent_iterations", len(logical_calls)))
        if exhausted is not None
        else len(logical_calls)
    )
    failed_attempts = tuple(item for item in attempts if item.outcome == "error")
    edit_iterations = tuple(
        item.metadata.get("agent_iteration")
        for item in readback.events
        if item.event_type == evidence.TOOL_EXECUTION_STARTED
        and item.metadata.get("resolved_name") in {"edit_file", "write_file"}
        and isinstance(item.metadata.get("agent_iteration"), int)
        and int(item.metadata["agent_iteration"]) > 0
    )

    metrics = RunMetrics(
        provider_logical_calls=available(len(logical_calls)),
        provider_attempts=available(len(attempts)),
        tool_calls_total=available(len(replayed.tool_executions)),
        unique_repo_files_read=read_values[0],
        total_repo_file_reads=read_values[1],
        repeated_repo_file_reads=read_values[2],
        skill_candidates_retrieved=available(
            sum(item.candidate_type is CandidateType.SKILL_CANDIDATE for item in retrieved_candidates)
        ),
        skills_referenced=available(skill_modes["referenced"]),
        skills_activated=available(skill_modes["activated"]),
        repeated_skill_reads=available(repeated_skill_reads),
        input_tokens=token_metric(input_tokens),
        output_tokens=token_metric(output_tokens),
        cached_tokens=(token_metric(cached_tokens) if cached_available else unavailable()),
        turn_latency_ms=latency,
        provider_retries=available(max(0, len(attempts) - len(logical_calls))),
        failed_tool_attempts=available(sum(item.outcome not in {None, "success"} for item in replayed.tool_executions)),
        validation_failures=available(sum(item.failure_stage == "validation" for item in replayed.tool_executions)),
        p3_approximate_context_tokens=(
            available(approximate_context_tokens) if approximate_available else unavailable()
        ),
        provider_failed_attempts=available(len(failed_attempts)),
        known_input_tokens_sum=token_metric(input_tokens),
        known_output_tokens_sum=token_metric(output_tokens),
        known_cached_tokens_sum=(token_metric(cached_tokens) if cached_available else unavailable()),
        attempts_with_usage=available(attempts_with_usage),
        attempts_without_usage=available(attempts_without_usage),
        token_accounting_status=token_status,
        agent_iterations=available(agent_iterations),
        first_edit_iteration=(
            available(min(edit_iterations)) if edit_iterations else unavailable()
        ),
        iteration_exhausted=exhausted is not None,
        final_synthesis_call_present=final_synthesis,
    )
    refs = {
        "retrieval_refs": retrieval_ids,
        "usage_refs": tuple(item.usage_id for item in usages),
        "retrieved_candidate_ids": tuple(sorted(retrieved_ids)),
        "candidate_ids": candidate_ids,
        "injected_ids": tuple(item.candidate_id for item in usages if item.usage_mode.value == "injected"),
        "referenced_ids": tuple(item.candidate_id for item in usages if item.usage_mode.value == "referenced"),
        "activated_ids": tuple(item.candidate_id for item in usages if item.usage_mode.value == "activated"),
        "normalized_provider_failure_categories": tuple(
            item.normalized_error_category
            or evidence.normalize_provider_failure(item.error_category)
            for item in failed_attempts
        ),
        "repository_read_paths": (
            tuple(repository_read_paths)
            if repository_read_paths is not None
            else evidence_paths
        ),
    }
    return metrics, refs


__all__ = ["available", "extract_run_metrics", "repeated_repository_reads", "unavailable"]
