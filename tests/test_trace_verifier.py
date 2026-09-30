from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from pico.tracing import evidence, replay, verifier


def _attempt(
    logical_call_id: str = "call-1",
    ordinal: int = 1,
    *,
    start: int = 2,
    complete: int | None = 3,
    outcome: str | None = "success",
    model: str = "model-a",
) -> evidence.ProviderAttemptEvidence:
    return evidence.ProviderAttemptEvidence(
        logical_call_id=logical_call_id,
        attempt_id=f"{logical_call_id}:attempt:{ordinal}",
        attempt_ordinal=ordinal,
        started_sequence=start,
        completed_sequence=complete,
        requested_model="model-a",
        attempted_model=model,
        actual_model=model if complete is not None else None,
        provider="fixture",
        outcome=outcome,
        finish_reason="stop" if outcome == "success" else "error" if outcome else None,
        error_category=None if outcome == "success" else "network" if outcome else None,
        request_digest="1" * 64,
        response_digest="2" * 64 if complete is not None else None,
        usage_available=complete is not None,
        usage={"prompt_tokens": 1} if complete is not None else {},
        duration_ms=1 if complete is not None else None,
        trace_id="trace-1",
        span_id="provider-span",
    )


def _provider_call(
    attempts: tuple[evidence.ProviderAttemptEvidence, ...] | None = None,
) -> replay.ProviderLogicalCall:
    values = attempts or (_attempt(),)
    sequences = [
        sequence
        for item in values
        for sequence in (item.started_sequence, item.completed_sequence)
        if sequence is not None
    ]
    return replay.ProviderLogicalCall(
        logical_call_id=values[0].logical_call_id,
        attempts=values,
        final_known_outcome=next(
            (item.outcome for item in reversed(values) if item.completed_sequence is not None),
            None,
        ),
        retry_count=max(0, len(values) - 1),
        fallback_count=sum(
            left.attempted_model != right.attempted_model
            for left, right in zip(values, values[1:], strict=False)
        ),
        usage=replay.ProviderUsageSummary(
            totals=(("prompt_tokens", sum(int(item.usage.get("prompt_tokens", 0)) for item in values)),),
            complete=all(item.complete for item in values),
            attempts_with_usage=sum(item.usage_available is True for item in values),
        ),
        first_sequence=min(sequences),
        last_sequence=max(sequences),
    )


def _tool(
    receipt_id: str,
    name: str,
    *,
    start: int,
    complete: int | None,
    model_call_id: str | None = None,
    parent_call_id: str | None = None,
    routed_via: str | None = None,
    effect: str | None = "read",
    outcome: str | None = "success",
    resolved_name: str | None | object = ...,
    failure_stage: str | None = None,
    failure_category: str | None = None,
) -> evidence.ToolExecutionEvidence:
    resolved = name if resolved_name is ... else resolved_name
    return evidence.ToolExecutionEvidence(
        receipt_id=receipt_id,
        model_call_id=model_call_id or f"model-{receipt_id}",
        parent_call_id=parent_call_id,
        started_sequence=start,
        completed_sequence=complete,
        requested_name=name,
        resolved_name=resolved,
        routed_via=routed_via,
        effect=effect,
        outcome=outcome,
        failure_stage=failure_stage,
        failure_category=failure_category,
        argument_digest="3" * 64,
        result_digest="4" * 64 if complete is not None else None,
        result_size=2 if complete is not None else None,
        duration_ms=1 if complete is not None else None,
        trace_id="trace-1",
        span_id="tool-span",
    )


def _make_replay(
    *,
    provider_calls: tuple[replay.ProviderLogicalCall, ...] | None = None,
    tools: tuple[evidence.ToolExecutionEvidence, ...] = (),
    runtime_outcome: str | None = "completed",
    delivery_outcomes: tuple[replay.DeliveryOutcome, ...] = (),
    session_ref: replay.SessionBoundaryReference | None = None,
    checkpoint_refs: tuple[replay.CheckpointReference, ...] = (),
    status: evidence.EvidenceCompleteness = evidence.EvidenceCompleteness.COMPLETE,
    warnings: tuple[str, ...] = (),
    inconsistencies: tuple[str, ...] = (),
    extra_events: tuple[tuple[int, str], ...] = (),
) -> replay.TraceReplayResult:
    calls = provider_calls if provider_calls is not None else (_provider_call(),)
    events: dict[int, str] = {1: evidence.TURN_STARTED}
    for call in calls:
        for attempt in call.attempts:
            if attempt.started_sequence is not None:
                events[attempt.started_sequence] = evidence.PROVIDER_ATTEMPT_STARTED
            if attempt.completed_sequence is not None:
                events[attempt.completed_sequence] = evidence.PROVIDER_ATTEMPT_COMPLETED
    for item in tools:
        if item.started_sequence is not None:
            events[item.started_sequence] = evidence.TOOL_EXECUTION_STARTED
        if item.completed_sequence is not None:
            events[item.completed_sequence] = evidence.TOOL_EXECUTION_COMPLETED
    for item in delivery_outcomes:
        events[item.sequence] = evidence.DELIVERY_OUTCOME
    for sequence, event_type in extra_events:
        events[sequence] = event_type
    terminal_sequence = max(events, default=0) + 1
    terminal = None
    if runtime_outcome is not None:
        events[terminal_sequence] = evidence.TURN_TERMINAL
        terminal = replay.RuntimeOutcome(
            sequence=terminal_sequence,
            outcome=runtime_outcome,
            lifecycle_event="TurnEnded" if runtime_outcome.startswith("completed") else "TurnFailed",
            error_class=None,
            provider_error_category=None,
            tool_calls=len([item for item in tools if item.parent_call_id is None]),
            tool_failures=sum(item.outcome == "failure" for item in tools),
        )

    timeline = tuple(
        replay.ReplayTimelineEntry(
            sequence=sequence,
            event_type=event_type,
            evidence_id=f"turn-1:{sequence}:{event_type}",
            trace_id="trace-1",
            span_id=f"span-{sequence}",
        )
        for sequence, event_type in sorted(events.items())
    )
    refs = tuple(
        replay.SourceEvidenceReference(
            sequence=item.sequence,
            event_type=item.event_type,
            evidence_id=item.evidence_id,
            structural_digest=f"{item.sequence % 10}" * 64,
        )
        for item in timeline
    )
    result = replay.TraceReplayResult(
        schema=replay.SCHEMA,
        schema_version=replay.SCHEMA_VERSION,
        turn_id="turn-1",
        trace_id="trace-1",
        conversation_id="test:chat",
        session_id=session_ref.session_id if session_ref else None,
        evidence_status=status,
        ordered_timeline=timeline,
        provider_calls=calls,
        tool_executions=tools,
        runtime_outcome=terminal,
        delivery_outcomes=delivery_outcomes,
        session_boundary_ref=session_ref,
        checkpoint_refs=checkpoint_refs,
        warnings=warnings,
        inconsistencies=inconsistencies,
        source_evidence_references=refs,
        replay_digest="",
    )
    return replace(result, replay_digest=replay.compute_replay_digest(result))


def _check(result: verifier.TraceVerificationResult, check_id: str) -> verifier.VerificationCheck:
    return next(item for item in result.checks if item.check_id == check_id)


def test_complete_valid_replay_passes_core_without_proving_task_success() -> None:
    result = verifier.verify_replay(_make_replay())

    assert result.overall_status is verifier.VerificationStatus.PASS
    assert result.derived_facts.runtime_success is True
    assert result.derived_facts.task_success_proven is False
    assert _check(result, "optional.delivery").status is verifier.VerificationStatus.INCONCLUSIVE
    with pytest.raises(FrozenInstanceError):
        result.turn_id = "changed"  # type: ignore[misc]


def test_partial_is_inconclusive_and_corrupt_is_failure() -> None:
    partial = verifier.verify_replay(
        _make_replay(status=evidence.EvidenceCompleteness.PARTIAL, warnings=("write_degradation",))
    )
    corrupt = verifier.verify_replay(
        _make_replay(
            status=evidence.EvidenceCompleteness.CORRUPT,
            inconsistencies=("duplicate_sequence",),
        )
    )

    assert partial.overall_status is verifier.VerificationStatus.INCONCLUSIVE
    assert corrupt.overall_status is verifier.VerificationStatus.FAIL


def test_agent_text_is_not_an_input_or_proof_of_verification() -> None:
    historical = _make_replay()
    result = verifier.verify_replay(historical)

    assert not hasattr(historical, "final_text")
    assert result.derived_facts.task_success_proven is False
    assert result.derived_facts.verification_after_last_mutation is None


def test_valid_provider_retry_chain_preserves_failure_then_success() -> None:
    attempts = (
        _attempt(ordinal=1, start=2, complete=3, outcome="error"),
        _attempt(ordinal=2, start=4, complete=5, outcome="success", model="model-b"),
    )
    result = verifier.verify_replay(_make_replay(provider_calls=(_provider_call(attempts),)))

    assert _check(result, "core.provider_receipts").status is verifier.VerificationStatus.PASS
    assert result.derived_facts.provider_attempt_count == 2
    assert result.derived_facts.provider_retry_count == 1
    assert result.derived_facts.provider_fallback_count == 1
    assert result.derived_facts.provider_recovered_after_failure is True


def test_invalid_provider_ordinal_fails_and_missing_completion_is_inconclusive() -> None:
    invalid = _attempt(ordinal=2)
    invalid_result = verifier.verify_replay(_make_replay(provider_calls=(_provider_call((invalid,)),)))
    incomplete = _attempt(complete=None, outcome=None)
    partial_result = verifier.verify_replay(
        _make_replay(
            provider_calls=(_provider_call((incomplete,)),),
            status=evidence.EvidenceCompleteness.PARTIAL,
            warnings=("incomplete_provider_attempt",),
        )
    )

    assert _check(invalid_result, "core.provider_receipts").status is verifier.VerificationStatus.FAIL
    assert _check(partial_result, "core.provider_receipts").status is verifier.VerificationStatus.INCONCLUSIVE
    assert partial_result.derived_facts.incomplete_provider_attempt_count == 1


def test_direct_and_meta_routed_tools_and_parent_relationship_are_valid() -> None:
    direct = _tool("direct", "read_file", start=4, complete=5)
    outer = _tool(
        "outer",
        "tool_call",
        start=6,
        complete=9,
        model_call_id="outer-call",
    )
    child = _tool(
        "child",
        "write_file",
        start=7,
        complete=8,
        model_call_id="child-call",
        parent_call_id="outer-call",
        routed_via="tool_call",
        effect="write",
    )
    result = verifier.verify_replay(_make_replay(tools=(direct, outer, child)))

    assert _check(result, "core.tool_receipts").status is verifier.VerificationStatus.PASS
    assert result.derived_facts.direct_tool_execution_count == 2
    assert result.derived_facts.meta_routed_tool_execution_count == 1
    assert result.derived_facts.mutation_observed is True


def test_requested_resolved_mismatch_and_impossible_parent_fail() -> None:
    mismatch = _tool("bad-name", "requested", start=4, complete=5, resolved_name="different")
    orphan = _tool(
        "orphan",
        "child",
        start=4,
        complete=5,
        parent_call_id="missing-parent",
        routed_via="tool_call",
    )

    mismatch_result = verifier.verify_replay(_make_replay(tools=(mismatch,)))
    orphan_result = verifier.verify_replay(_make_replay(tools=(orphan,)))

    assert _check(mismatch_result, "core.tool_receipts").status is verifier.VerificationStatus.FAIL
    assert _check(orphan_result, "core.tool_receipts").status is verifier.VerificationStatus.FAIL


def test_tool_failures_and_unknown_effectful_completion_remain_distinct() -> None:
    schema = _tool(
        "schema",
        "write_file",
        start=4,
        complete=5,
        effect="write",
        outcome="failure",
        failure_stage="schema_validation",
        failure_category="invalid_arguments",
    )
    timeout = _tool(
        "timeout",
        "remote",
        start=6,
        complete=7,
        effect="external",
        outcome="failure",
        failure_stage="timeout",
        failure_category="timeout",
    )
    unknown = _tool(
        "unknown",
        "remote_write",
        start=8,
        complete=None,
        effect="external",
        outcome=None,
    )
    result = verifier.verify_replay(
        _make_replay(
            tools=(schema, timeout, unknown),
            status=evidence.EvidenceCompleteness.PARTIAL,
            warnings=("incomplete_tool_execution", "unsafe_tool_completion_unknown:unknown"),
        )
    )

    assert result.overall_status is verifier.VerificationStatus.INCONCLUSIVE
    assert result.derived_facts.validation_failure_count == 1
    assert result.derived_facts.timeout_count == 1
    assert result.derived_facts.incomplete_tool_execution_count == 1
    assert result.derived_facts.external_effect_observed is False
    assert result.derived_facts.external_state_current is None


def test_explicit_verification_after_mutation_is_proven_only_by_external_profile() -> None:
    mutation = _tool("mutation", "write_file", start=4, complete=5, effect="write")
    verification = _tool("verify", "frozen_verifier", start=6, complete=7, effect="execute")
    historical = _make_replay(tools=(mutation, verification))

    default = verifier.verify_replay(historical)
    explicit = verifier.verify_replay(
        historical,
        verifier.TraceVerificationProfile(
            profile_id="fixture",
            verification_receipt_ids=("verify",),
            require_verification_after_mutation=True,
        ),
    )

    assert default.derived_facts.verification_after_last_mutation is None
    assert explicit.derived_facts.verification_after_last_mutation is True
    assert _check(explicit, "optional.verification_after_mutation").status is verifier.VerificationStatus.PASS
    assert explicit.overall_status is verifier.VerificationStatus.PASS


def test_required_verification_without_explicit_evidence_is_inconclusive() -> None:
    mutation = _tool("mutation", "write_file", start=4, complete=5, effect="write")
    result = verifier.verify_replay(
        _make_replay(tools=(mutation,)),
        verifier.TraceVerificationProfile(require_verification_after_mutation=True),
    )

    assert result.overall_status is verifier.VerificationStatus.INCONCLUSIVE
    assert result.derived_facts.verification_after_last_mutation is None


def test_duplicate_receipt_fails_but_repeated_intent_is_not_runtime_duplication() -> None:
    duplicate_left = _tool("same", "read_file", start=4, complete=5, model_call_id="model-a")
    duplicate_right = _tool("same", "read_file", start=6, complete=7, model_call_id="model-b")
    repeated_left = _tool("one", "read_file", start=4, complete=5, model_call_id="model-a")
    repeated_right = _tool("two", "read_file", start=6, complete=7, model_call_id="model-b")

    duplicate = verifier.verify_replay(_make_replay(tools=(duplicate_left, duplicate_right)))
    repeated = verifier.verify_replay(_make_replay(tools=(repeated_left, repeated_right)))

    assert _check(duplicate, "core.tool_receipts").status is verifier.VerificationStatus.FAIL
    assert _check(repeated, "core.tool_receipts").status is verifier.VerificationStatus.PASS
    assert repeated.derived_facts.repeated_resolved_executions == ("read_file",)
    assert repeated.derived_facts.duplicate_model_call_ids == ()


def test_duplicate_model_call_identity_is_inconclusive_not_asserted_runtime_duplication() -> None:
    left = _tool("one", "read_file", start=4, complete=5, model_call_id="same-model-call")
    right = _tool("two", "read_file", start=6, complete=7, model_call_id="same-model-call")

    result = verifier.verify_replay(_make_replay(tools=(left, right)))

    assert _check(result, "core.tool_receipts").status is verifier.VerificationStatus.INCONCLUSIVE
    assert result.derived_facts.duplicate_model_call_ids == ("same-model-call",)


@pytest.mark.parametrize(
    ("runtime_outcome", "delivery_outcome", "runtime_fact", "delivery_fact"),
    [
        ("completed", "dropped", "runtime_success", "delivery_dropped"),
        ("error", "delivered", "runtime_failure", "delivery_success"),
    ],
)
def test_runtime_and_delivery_facts_remain_independent(
    runtime_outcome: str,
    delivery_outcome: str,
    runtime_fact: str,
    delivery_fact: str,
) -> None:
    delivery = replay.DeliveryOutcome(
        sequence=4,
        channel="test",
        event="Text",
        outcome=delivery_outcome,
        attempts=1,
        error_class=None,
        delivery_trace_id="delivery-trace",
    )
    result = verifier.verify_replay(
        _make_replay(runtime_outcome=runtime_outcome, delivery_outcomes=(delivery,))
    )

    assert result.overall_status is verifier.VerificationStatus.PASS
    assert getattr(result.derived_facts, runtime_fact) is True
    assert getattr(result.derived_facts, delivery_fact) is True
    if delivery_outcome == "delivered":
        assert result.derived_facts.runtime_success is False


def test_session_and_checkpoint_match_or_mismatch_are_distinguished() -> None:
    boundary = replay.SessionBoundaryReference(
        session_id="test:chat",
        boundary_id="boundary-1",
        turn_id="turn-1",
        version=1,
        message_count=3,
        history_digest="history",
        resolved=True,
    )
    checkpoint = replay.CheckpointReference(
        record_id="record-1",
        checkpoint_id="tree-1",
        revision=1,
        status="created",
        schema_version=1,
        session_id="test:chat",
        boundary_id="boundary-1",
        turn_id="turn-1",
        resolved=True,
    )
    matched = verifier.verify_replay(_make_replay(session_ref=boundary, checkpoint_refs=(checkpoint,)))
    mismatched = verifier.verify_replay(
        _make_replay(
            session_ref=boundary,
            checkpoint_refs=(replace(checkpoint, boundary_id="other"),),
        )
    )
    foreign_boundary = verifier.verify_replay(
        _make_replay(session_ref=replace(boundary, turn_id="foreign-turn"))
    )
    foreign_checkpoint = verifier.verify_replay(
        _make_replay(
            session_ref=boundary,
            checkpoint_refs=(replace(checkpoint, turn_id="foreign-turn"),),
        )
    )

    assert _check(matched, "core.session_reference").status is verifier.VerificationStatus.PASS
    assert _check(matched, "core.checkpoint_references").status is verifier.VerificationStatus.PASS
    assert _check(mismatched, "core.checkpoint_references").status is verifier.VerificationStatus.FAIL
    assert _check(foreign_boundary, "core.session_reference").status is verifier.VerificationStatus.FAIL
    assert _check(foreign_checkpoint, "core.checkpoint_references").status is verifier.VerificationStatus.FAIL


def test_unresolved_optional_reference_is_inconclusive() -> None:
    boundary = replay.SessionBoundaryReference(
        session_id="test:chat",
        boundary_id="boundary-1",
        turn_id="turn-1",
        version=1,
        message_count=3,
        history_digest="history",
        resolved=False,
    )
    result = verifier.verify_replay(_make_replay(session_ref=boundary))

    assert _check(result, "core.session_reference").status is verifier.VerificationStatus.INCONCLUSIVE
    assert _check(result, "optional.checkpoint_references").status is verifier.VerificationStatus.INCONCLUSIVE
    assert result.overall_status is verifier.VerificationStatus.INCONCLUSIVE


def test_sequence_gap_duplicate_and_conflicting_terminal_fail_closed() -> None:
    base = _make_replay()
    gap_timeline = tuple(item for item in base.ordered_timeline if item.sequence != 2)
    gap_refs = tuple(item for item in base.source_evidence_references if item.sequence != 2)
    gap = replace(base, ordered_timeline=gap_timeline, source_evidence_references=gap_refs, replay_digest="")
    gap = replace(gap, replay_digest=replay.compute_replay_digest(gap))

    duplicated_timeline = (*base.ordered_timeline, replace(base.ordered_timeline[-1], evidence_id="turn-1:dup"))
    duplicate = replace(base, ordered_timeline=duplicated_timeline, replay_digest="")
    duplicate = replace(duplicate, replay_digest=replay.compute_replay_digest(duplicate))

    conflict = _make_replay(
        runtime_outcome=None,
        status=evidence.EvidenceCompleteness.CORRUPT,
        inconsistencies=("conflicting_terminal_outcomes",),
        extra_events=((4, evidence.TURN_TERMINAL), (5, evidence.TURN_TERMINAL)),
    )

    assert _check(verifier.verify_replay(gap), "core.sequence").status is verifier.VerificationStatus.INCONCLUSIVE
    assert verifier.verify_replay(duplicate).overall_status is verifier.VerificationStatus.FAIL
    assert verifier.verify_replay(conflict).overall_status is verifier.VerificationStatus.FAIL


def test_altered_replay_or_source_digest_is_detected() -> None:
    base = _make_replay()
    altered_replay = replace(base, replay_digest="0" * 64)
    altered_ref = replace(base.source_evidence_references[0], structural_digest="not-a-digest")
    altered_source = replace(base, source_evidence_references=(altered_ref, *base.source_evidence_references[1:]), replay_digest="")
    altered_source = replace(altered_source, replay_digest=replay.compute_replay_digest(altered_source))

    assert _check(verifier.verify_replay(altered_replay), "core.replay_integrity").status is verifier.VerificationStatus.FAIL
    assert _check(verifier.verify_replay(altered_source), "core.provenance").status is verifier.VerificationStatus.FAIL


def test_verification_digest_is_deterministic_and_bound_to_profile() -> None:
    historical = _make_replay()
    first = verifier.verify_replay(historical)
    second = verifier.verify_replay(historical)
    other_profile = verifier.verify_replay(
        historical,
        verifier.TraceVerificationProfile(profile_id="other"),
    )
    configured_profile = verifier.verify_replay(
        historical,
        verifier.TraceVerificationProfile(verification_receipt_ids=("receipt",)),
    )

    assert first.verification_digest == second.verification_digest
    assert first.verification_digest != other_profile.verification_digest
    assert first.verification_digest != configured_profile.verification_digest
    assert verifier.compute_verification_digest(first) == first.verification_digest


def test_verifier_is_side_effect_free_and_requires_no_live_services() -> None:
    result = verifier.verify_replay(_make_replay())

    assert result.overall_status is verifier.VerificationStatus.PASS
    assert not hasattr(verifier, "Provider")
    assert not hasattr(verifier, "ToolRegistry")
    assert not hasattr(verifier, "DeliveryHub")
    assert not hasattr(verifier, "SessionManager")
    assert not hasattr(verifier, "WorkspaceRestoreService")
