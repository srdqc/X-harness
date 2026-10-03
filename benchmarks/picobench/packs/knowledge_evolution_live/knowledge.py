"""Preparation of the compact reviewed P3R corpus through public P3 APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest
from pico.knowledge_evolution import (
    CandidateComparison,
    CandidateEligibilityInput,
    CandidateRelation,
    CandidateType,
    ContentClass,
    ExtractionStatus,
    KnowledgeExtractionRequest,
    KnowledgeLifecycleManager,
    KnowledgeProposal,
    KnowledgeRecordStore,
    LifecycleActorType,
    LifecycleReason,
    LifecycleState,
    RepositoryScopeResolver,
    ReviewDecision,
    ReviewerType,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
    create_review,
    evaluate_candidate_eligibility,
    extract_candidates,
    materialize_candidate,
    validate_candidate,
)
from pico.tracing import evidence, replay, verifier
from pico.tracing.store import TraceStore

_NOW = "2026-10-02T00:00:00Z"


@dataclass(frozen=True)
class KnowledgeCorpusItem:
    corpus_id: str
    proposal: KnowledgeProposal


CORPUS = (
    KnowledgeCorpusItem("immutable-record-conventions", KnowledgeProposal(CandidateType.MEMORY_FACT.value, ContentClass.FACT.value, "Immutable evidence conventions", "Durable evidence records use canonical serialization, structural digests, immutable create-or-compare writes, and validated reloads.", ("The change introduces or reads durable evidence",), ("tool:read_file",), fact_subject="durable evidence convention", fact_value="canonical digest and immutable write")),
    KnowledgeCorpusItem("phase-test-registration", KnowledgeProposal(CandidateType.SKILL_CANDIDATE.value, ContentClass.PROCEDURE.value, "Register focused phase tests", "Inspect the canonical test matrix, add the narrowest reusable suite, and validate deduplicated target resolution before broader acceptance.", ("The repository uses scripts/run_tests.py",), ("tool:read_file",), validation_expectations=("Focused tests and matrix-resolution tests pass",))),
    KnowledgeCorpusItem("scoped-skill-flow", KnowledgeProposal(CandidateType.MEMORY_FACT.value, ContentClass.FACT.value, "Scoped Skill flow", "Repository-scoped Skills remain gated through SkillRegistry, Router, Resolver, and Skills context; retrieval does not bypass those owners.", ("The task touches P3 Skill reuse",), ("tool:read_file",), fact_subject="p3 skill flow", fact_value="registry router resolver context")),
    KnowledgeCorpusItem("structured-evidence-first", KnowledgeProposal(CandidateType.EXPERIENCE.value, ContentClass.STRATEGY.value, "Prefer structured Runtime evidence", "Measure Provider, Tool, replay, and knowledge activity from durable structured receipts rather than console logs or Agent prose.", ("Structured Runtime evidence is available",), ("tool:read_file",))),
    KnowledgeCorpusItem("fail-closed-recovery", KnowledgeProposal(CandidateType.EXPERIENCE.value, ContentClass.RECOVERY.value, "Recover without weakening trust gates", "When a schema or applicability edge case fails, preserve fail-closed behavior, locate the owning boundary, and add a deterministic regression before changing semantics.", ("A deterministic validation or applicability failure is reproducible",), ("tool:read_file",))),
    KnowledgeCorpusItem("portable-artifact-paths", KnowledgeProposal(CandidateType.EXPERIENCE.value, ContentClass.WARNING.value, "Keep physical benchmark paths short", "Keep Windows experiment paths short while retaining full logical identities and digests in immutable evidence.", ("The benchmark creates nested temporary artifacts",), ("tool:read_file",))),
)


class _SingleProposalExtractor:
    extractor_id = "p3r-frozen-corpus"
    extractor_version = "1"

    def __init__(self, proposal: KnowledgeProposal) -> None:
        self.proposal = proposal

    def extract(self, context):
        del context
        return (self.proposal,)


def corpus_digest() -> str:
    return canonical_digest(CORPUS)


def prepare_approved_corpus(
    *,
    state_root: Path,
    workspace: Path,
    reviewer_id: str,
) -> tuple[str, ...]:
    """Build ACTIVE materialized knowledge; never concatenate it into prompts."""

    store = KnowledgeRecordStore(state_root)
    scope_result = RepositoryScopeResolver(workspace, state_root).resolve()
    if not scope_result.resolved or scope_result.identity is None:
        raise RuntimeError("P3R requires a resolved repository scope")
    scope = scope_result.identity
    candidate_ids: list[str] = []
    for index, item in enumerate(CORPUS):
        turn_id = f"p3r-source-{index}"
        trace_root = state_root / "p3r-source-traces"
        recorder = evidence.TurnEvidenceRecorder(
            turn_id=turn_id,
            conversation_id=f"p3r:source:{index}",
            trace_id=f"p3r-source-trace-{index}",
            root_span_id=f"p3r-source-span-{index}",
            writer=TraceStore(trace_root).append_event,
        )
        recorder.emit(evidence.TURN_STARTED)
        recorder.emit(evidence.TURN_TERMINAL, metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"})
        replay_result = replay.replay_turn(trace_root, turn_id)
        verification = verifier.verify_replay(replay_result)
        target_digest = canonical_digest({"corpus_id": item.corpus_id, "state": "verified"})
        success = TaskSuccessEvidence.create(
            evidence_id=f"p3r-source-success-{index}",
            source_kind=TaskSuccessSource.FROZEN_BENCHMARK,
            source_turn_id=turn_id,
            status=TaskSuccessStatus.PASS,
            producer_id="p3r-sealed-corpus-verifier",
            producer_version="1",
            repository_scope_id=scope.repository_scope_id,
            target_state_digest=target_digest,
            result_digest=canonical_digest({"corpus_id": item.corpus_id, "result": "pass"}),
            provenance_refs=(f"p3r-corpus:{item.corpus_id}",),
            created_at=_NOW,
        )
        store.write_task_success(success)
        eligibility_input = CandidateEligibilityInput(
            turn_id=turn_id,
            replay=replay_result,
            verification=verification,
            task_success=(success,),
            scope=scope_result,
            resulting_state_digest=target_digest,
            extraction_policy="p3.manual-candidate",
            extraction_policy_version=1,
            candidate_type=CandidateType(item.proposal.candidate_type),
            provenance_turn_ids=(turn_id,),
        )
        eligibility = evaluate_candidate_eligibility(eligibility_input)
        extraction = extract_candidates(
            KnowledgeExtractionRequest(f"p3r-extraction-{index}", eligibility_input, eligibility),
            _SingleProposalExtractor(item.proposal),
            store,
            created_at=_NOW,
            created_by="extractor:p3r-corpus",
        )
        if extraction.status is not ExtractionStatus.COMPLETED or len(extraction.candidate_outcomes) != 1:
            raise RuntimeError(f"P3R corpus extraction failed: {item.corpus_id}")
        outcome = extraction.candidate_outcomes[0]
        candidate = store.read_candidate(outcome.candidate_id)
        if candidate is None:
            raise RuntimeError("extracted P3R candidate is missing")
        comparison = CandidateComparison(
            CandidateRelation(outcome.relation),
            candidate.candidate_id,
            outcome.existing_candidate_ids,
            outcome.compared_candidate_count,
        )
        validation = validate_candidate(
            store,
            candidate_id=candidate.candidate_id,
            comparison=comparison,
            expected_repository_scope_id=scope.repository_scope_id,
            validation_id=f"p3r-validation-{index}",
            validator_id="p3r-technical-validator",
            validator_version="1",
            validated_at=_NOW,
        )
        manager = KnowledgeLifecycleManager(store)
        for transition_id, source, target, reason in (
            (f"p3r-eligible-{index}", LifecycleState.EXTRACTED, LifecycleState.ELIGIBLE, LifecycleReason.ELIGIBILITY_CONFIRMED),
            (f"p3r-validated-{index}", LifecycleState.ELIGIBLE, LifecycleState.VALIDATED, LifecycleReason.TECHNICAL_VALIDATION_PASSED),
            (f"p3r-pending-{index}", LifecycleState.VALIDATED, LifecycleState.PENDING_REVIEW, LifecycleReason.SUBMITTED_FOR_REVIEW),
        ):
            manager.transition(
                transition_id=transition_id,
                candidate_id=candidate.candidate_id,
                from_state=source,
                to_state=target,
                actor_type=LifecycleActorType.SYSTEM,
                actor_id="p3r",
                reason=reason,
                transitioned_at=_NOW,
                evidence_refs=(("eligibility:" + eligibility.result_digest,) if target is LifecycleState.ELIGIBLE else ()),
                validation=(validation if target in {LifecycleState.VALIDATED, LifecycleState.PENDING_REVIEW} else None),
            )
        review = create_review(
            store,
            review_id=f"p3r-review-{index}",
            candidate_id=candidate.candidate_id,
            validation=validation,
            comparison=comparison,
            reviewer_type=ReviewerType.HUMAN,
            reviewer_id=reviewer_id,
            decision=ReviewDecision.APPROVE,
            reason="Approved as bounded non-oracle P3R repository guidance.",
            reviewed_at=_NOW,
        )
        manager.transition(
            transition_id=f"p3r-active-{index}",
            candidate_id=candidate.candidate_id,
            from_state=LifecycleState.PENDING_REVIEW,
            to_state=LifecycleState.ACTIVE,
            actor_type=LifecycleActorType.HUMAN,
            actor_id=reviewer_id,
            reason=LifecycleReason.HUMAN_APPROVED,
            transitioned_at=_NOW,
            validation=validation,
            review=review,
        )
        materialize_candidate(
            store,
            materialization_id=f"p3r-materialization-{index}",
            candidate_id=candidate.candidate_id,
            validation=validation,
            review=review,
        )
        candidate_ids.append(candidate.candidate_id)
    return tuple(candidate_ids)


__all__ = ["CORPUS", "KnowledgeCorpusItem", "corpus_digest", "prepare_approved_corpus"]
