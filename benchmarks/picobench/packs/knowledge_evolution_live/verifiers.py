"""Sealed post-Turn verifiers kept outside evaluated worktrees."""

from __future__ import annotations

import ast
import fnmatch
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest


@dataclass(frozen=True)
class HiddenVerifierSpec:
    verifier_id: str
    required_changed_paths: tuple[str, ...]
    required_source_markers: tuple[tuple[str, str], ...]
    targeted_tests: tuple[str, ...]
    required_changed_path_alternatives: tuple[tuple[str, ...], ...] = ()
    version: int = 2

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class HiddenVerifierResult:
    passed: bool
    findings: tuple[str, ...]
    verifier_id: str
    verifier_digest: str
    infrastructure_failure: bool = False

    @property
    def safety_passed(self) -> bool:
        return "prohibited_change_detected" not in self.findings


@dataclass(frozen=True)
class OfficialVerifierSpec:
    verifier_id: str
    required_changed_paths: tuple[str, ...]
    targeted_tests: tuple[str, ...]
    semantic_probe: str
    required_source_markers: tuple[tuple[str, str], ...] = ()
    required_changed_path_alternatives: tuple[tuple[str, ...], ...] = ()
    version: int = 1

    @property
    def digest(self) -> str:
        return canonical_digest(self)


_SPECS = (
    HiddenVerifierSpec(
        "p3r-v-nav-retrieval-list",
        ("pico/knowledge_evolution/store.py", "tests/*"),
        (("pico/knowledge_evolution/store.py", "list_retrievals"),),
        ("tests/test_knowledge_retrieval.py",),
    ),
    HiddenVerifierSpec(
        "p3r-v-nav-association-list",
        ("pico/knowledge_evolution/store.py", "tests/*"),
        (("pico/knowledge_evolution/store.py", "list_outcome_associations"),),
        ("tests/test_knowledge_usage.py",),
    ),
    HiddenVerifierSpec(
        "p3r-v-nav-campaign-matrix",
        ("scripts/testing/test_matrix.json", "tests/*"),
        (("scripts/testing/test_matrix.json", "p3r"),),
        ("tests/test_test_runner.py",),
    ),
    HiddenVerifierSpec(
        "p3r-v-impl-retrieval-summary",
        ("pico/knowledge_evolution/retrieval.py", "tests/*"),
        (("pico/knowledge_evolution/retrieval.py", "selected_count"),),
        ("tests/test_knowledge_retrieval.py",),
    ),
    HiddenVerifierSpec(
        "p3r-v-impl-usage-summary",
        ("pico/knowledge_evolution/usage.py", "tests/*"),
        (("pico/knowledge_evolution/usage.py", "KnowledgeUsage"),),
        ("tests/test_knowledge_usage.py",),
    ),
    HiddenVerifierSpec(
        "p3r-v-impl-report-reader",
        ("benchmarks/picobench/packs/knowledge_evolution/*", "tests/*"),
        (("benchmarks/picobench/packs/knowledge_evolution/benchmark.py", "read_"),),
        ("tests/test_picobench_knowledge_evolution.py",),
    ),
    HiddenVerifierSpec(
        "p3r-v-debug-guard",
        ("pico/knowledge_evolution/canonicalize.py", "tests/*"),
        (),
        ("tests/test_knowledge_canonicalization.py", "tests/test_knowledge_applicability.py"),
        version=3,
    ),
    HiddenVerifierSpec(
        "p3r-v-debug-association",
        ("pico/knowledge_evolution/usage.py", "tests/*"),
        (("pico/knowledge_evolution/usage.py", "task-success Turn"),),
        ("tests/test_knowledge_usage.py",),
    ),
    HiddenVerifierSpec(
        "p3r-v-debug-path",
        ("benchmarks/picobench/packs/knowledge_evolution/*", "tests/*"),
        (("benchmarks/picobench/packs/knowledge_evolution/benchmark.py", "Windows"),),
        ("tests/test_picobench_knowledge_evolution.py",),
    ),
    HiddenVerifierSpec(
        "p3r-v-integration-skill",
        (),
        (("pico/memory_engine/skill_forge/knowledge_source.py", "ApplicableKnowledgeSkillSource"),),
        ("tests/test_knowledge_runtime_integration.py",),
        required_changed_path_alternatives=(
            ("pico/memory_engine/skill_forge/*", "tests/*"),
            ("tests/test_knowledge_runtime_integration.py",),
        ),
    ),
    HiddenVerifierSpec(
        "p3r-v-integration-trace",
        ("pico/tracing/*", "tests/*"),
        (("pico/tracing/replay.py", "knowledge"),),
        ("tests/test_trace_replay.py", "tests/test_knowledge_runtime_integration.py"),
    ),
    HiddenVerifierSpec(
        "p3r-v-integration-suite",
        ("scripts/testing/test_matrix.json", "tests/*"),
        (("scripts/testing/test_matrix.json", "picobench"),),
        ("tests/test_test_runner.py", "tests/test_picobench_knowledge_evolution.py"),
    ),
)

_RETRIEVAL_PRELUDE = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from pico.knowledge_evolution import KnowledgeRecordStore, KnowledgeRetrievalReceipt
def receipt(identity, turn):
    return KnowledgeRetrievalReceipt.create(
        retrieval_id=identity, turn_id=turn, repository_scope_id="a"*64,
        query_digest="b"*64, candidate_set_digest="c"*64,
        ranked_candidate_ids=("candidate-1",), selected_candidate_ids=("candidate-1",),
        suppressed=(), applicable_count=1, suppressed_count=0, latency_ms=0.0,
        created_at="2026-10-03T00:00:00Z", retrieval_policy="probe", retrieval_version=1,
    )
"""

_USAGE_PRELUDE = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from pico.knowledge_evolution import CandidateType, KnowledgeRecordStore, KnowledgeUsageMode, KnowledgeUsageReceipt
def usage(identity, turn="turn-1", candidate="candidate-1", mode=KnowledgeUsageMode.INJECTED, score=1.0):
    return KnowledgeUsageReceipt.create(
        usage_id=identity, turn_id=turn, repository_scope_id="a"*64,
        candidate_id=candidate, candidate_manifest_digest="b"*64,
        materialization_digest="c"*64, knowledge_type=CandidateType.EXPERIENCE,
        lifecycle_state="active", applicability_id="app-1", applicability_digest="d"*64,
        retrieval_id="retrieval-1", retrieval_rank=1, retrieval_score=score,
        usage_mode=mode, context_identity="context:test", created_at="2026-10-03T00:00:00Z",
    )
"""

_OFFICIAL_SPECS = (
    OfficialVerifierSpec(
        "p3r4-v-nav-retrieval-list",
        ("pico/knowledge_evolution/store.py", "tests/*"),
        ("tests/test_knowledge_retrieval.py",),
        _RETRIEVAL_PRELUDE
        + r"""
with TemporaryDirectory() as root:
    store=KnowledgeRecordStore(Path(root)); store.write_retrieval(receipt("r-b","turn-b")); store.write_retrieval(receipt("r-a","turn-a"))
    assert tuple(x.retrieval_id for x in store.list_retrievals()) == ("r-a","r-b")
    assert tuple(x.retrieval_id for x in store.list_retrievals(turn_id="turn-b")) == ("r-b",)
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-nav-outcome-list",
        ("pico/knowledge_evolution/store.py", "tests/*"),
        ("tests/test_knowledge_usage.py",),
        r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from pico.knowledge_evolution import KnowledgeRecordStore, KnowledgeUsageOutcomeAssociation, TaskSuccessStatus
def item(identity, turn):
    return KnowledgeUsageOutcomeAssociation.create(association_id=identity, turn_id=turn, usage_ids=("u",), usage_digests=("a"*64,), candidate_ids=("c",), task_success_evidence_id="e", task_success_evidence_digest="b"*64, task_success_status=TaskSuccessStatus.PASS, association_policy="probe", association_version=1, created_at="2026-10-03T00:00:00Z")
with TemporaryDirectory() as root:
    store=KnowledgeRecordStore(Path(root)); store.write_outcome_association(item("o-b","turn-b")); store.write_outcome_association(item("o-a","turn-a"))
    assert tuple(x.association_id for x in store.list_outcome_associations()) == ("o-a","o-b")
    assert tuple(x.association_id for x in store.list_outcome_associations(turn_id="turn-b")) == ("o-b",)
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-nav-applicability-list",
        ("pico/knowledge_evolution/store.py", "tests/*"),
        ("tests/test_knowledge_applicability.py",),
        r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from pico.knowledge_evolution import ApplicabilityReason, ApplicabilityStatus, KnowledgeApplicabilityResult, KnowledgeRecordStore, LifecycleState
def item(identity, candidate, status):
    reason=ApplicabilityReason.APPLICABLE if status is ApplicabilityStatus.APPLICABLE else ApplicabilityReason.NOT_ACTIVE
    return KnowledgeApplicabilityResult.create(applicability_id=identity,candidate_id=candidate,candidate_manifest_digest="a"*64,repository_scope_id="b"*64,lifecycle_state=LifecycleState.ACTIVE,lifecycle_transition_digest="c"*64,materialization_id="m",materialization_digest="d"*64,status=status,reasons=(reason,),guard_evidence=(),evaluated_at="2026-10-03T00:00:00Z",evaluator_policy="probe",evaluator_version=1)
with TemporaryDirectory() as root:
    store=KnowledgeRecordStore(Path(root)); store.write_applicability(item("a-b","c-b",ApplicabilityStatus.NOT_APPLICABLE)); store.write_applicability(item("a-a","c-a",ApplicabilityStatus.APPLICABLE))
    assert tuple(x.applicability_id for x in store.list_applicability_results()) == ("a-a","a-b")
    assert tuple(x.applicability_id for x in store.list_applicability_results(candidate_id="c-b")) == ("a-b",)
    assert tuple(x.applicability_id for x in store.list_applicability_results(status=ApplicabilityStatus.APPLICABLE)) == ("a-a",)
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v2-nav-applicability-list",
        ("pico/knowledge_evolution/store.py", "tests/*"),
        ("tests/test_knowledge_applicability.py",),
        r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from pico.knowledge_evolution import ApplicabilityReason, ApplicabilityStatus, KnowledgeApplicabilityResult, KnowledgeRecordStore, KnowledgeStoreError, LifecycleState
def item(identity, candidate, status):
    reason=ApplicabilityReason.APPLICABLE if status is ApplicabilityStatus.APPLICABLE else ApplicabilityReason.NOT_ACTIVE
    return KnowledgeApplicabilityResult.create(applicability_id=identity,candidate_id=candidate,candidate_manifest_digest="a"*64,repository_scope_id="b"*64,lifecycle_state=LifecycleState.ACTIVE,lifecycle_transition_digest="c"*64,materialization_id="m",materialization_digest="d"*64,status=status,reasons=(reason,),guard_evidence=(),evaluated_at="2026-10-03T00:00:00Z",evaluator_policy="probe",evaluator_version=1)
with TemporaryDirectory() as root:
    store=KnowledgeRecordStore(Path(root))
    assert store.list_applicability_results() == ()
    store.write_applicability(item("a-c","c-a",ApplicabilityStatus.NOT_APPLICABLE))
    store.write_applicability(item("a-b","c-b",ApplicabilityStatus.NOT_APPLICABLE))
    store.write_applicability(item("a-a","c-a",ApplicabilityStatus.APPLICABLE))
    assert tuple(x.applicability_id for x in store.list_applicability_results()) == ("a-a","a-b","a-c")
    assert tuple(x.applicability_id for x in store.list_applicability_results(candidate_id="c-b")) == ("a-b",)
    assert tuple(x.applicability_id for x in store.list_applicability_results(status=ApplicabilityStatus.APPLICABLE)) == ("a-a",)
    assert tuple(x.applicability_id for x in store.list_applicability_results(candidate_id="c-a",status=ApplicabilityStatus.NOT_APPLICABLE)) == ("a-c",)
    assert store.list_applicability_results(candidate_id="missing") == ()
    reloaded=KnowledgeRecordStore(Path(root))
    assert tuple(x.applicability_id for x in reloaded.list_applicability_results()) == ("a-a","a-b","a-c")
    original=store.applicability_results / "a-a.json"
    original.rename(store.applicability_results / "mismatched.json")
    try: store.list_applicability_results()
    except KnowledgeStoreError: pass
    else: raise AssertionError("record/path identity mismatch accepted")
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-impl-retrieval-summary",
        ("pico/knowledge_evolution/retrieval.py", "tests/*"),
        ("tests/test_knowledge_retrieval.py",),
        _RETRIEVAL_PRELUDE
        + r"""
r=receipt("r","turn")
summary=r.selection_summary
assert summary["ranked"] == 1 and summary["selected"] == 1 and summary["suppressed"] == 0
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-impl-usage-visibility",
        ("pico/knowledge_evolution/usage.py", "tests/*"),
        ("tests/test_knowledge_usage.py",),
        _USAGE_PRELUDE
        + r"""
assert usage("u-r", mode=KnowledgeUsageMode.RETRIEVED).is_model_visible is False
for mode in (KnowledgeUsageMode.INJECTED, KnowledgeUsageMode.REFERENCED, KnowledgeUsageMode.ACTIVATED):
    assert usage("u-"+mode.value, mode=mode).is_model_visible is True
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-impl-outcome-counts",
        ("pico/knowledge_evolution/usage.py", "tests/*"),
        ("tests/test_knowledge_usage.py",),
        r"""
from pico.knowledge_evolution import KnowledgeUsageOutcomeAssociation, TaskSuccessStatus
r=KnowledgeUsageOutcomeAssociation.create(association_id="o",turn_id="t",usage_ids=("u1","u2"),usage_digests=("a"*64,"b"*64),candidate_ids=("c1",),task_success_evidence_id="e",task_success_evidence_digest="c"*64,task_success_status=TaskSuccessStatus.PASS,association_policy="probe",association_version=1,created_at="2026-10-03T00:00:00Z")
assert r.usage_count == 2 and r.candidate_count == 1
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-debug-duplicate-usage",
        ("pico/knowledge_evolution/usage.py", "tests/*"),
        ("tests/test_knowledge_usage.py",),
        r"""
from pico.knowledge_evolution import KnowledgeUsageOutcomeAssociation, TaskSuccessStatus
try:
    KnowledgeUsageOutcomeAssociation.create(association_id="o",turn_id="t",usage_ids=("u","u"),usage_digests=("a"*64,"a"*64),candidate_ids=("c",),task_success_evidence_id="e",task_success_evidence_digest="b"*64,task_success_status=TaskSuccessStatus.PASS,association_policy="probe",association_version=1,created_at="2026-10-03T00:00:00Z")
except ValueError: pass
else: raise AssertionError("duplicate usage IDs accepted")
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-debug-retrieval-membership",
        ("pico/knowledge_evolution/retrieval.py", "tests/*"),
        ("tests/test_knowledge_retrieval.py",),
        _RETRIEVAL_PRELUDE
        + r"""
base=receipt("r","t").to_dict()
for selected in (("missing",), ("candidate-1","candidate-1")):
    value=dict(base); value["selected_candidate_ids"]=selected
    value.pop("retrieval_digest")
    try: KnowledgeRetrievalReceipt.create(**value)
    except ValueError: pass
    else: raise AssertionError("invalid selected set accepted")
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-debug-finite-score",
        ("pico/knowledge_evolution/usage.py", "tests/*"),
        ("tests/test_knowledge_usage.py",),
        _USAGE_PRELUDE
        + r"""
for score in (float("nan"), float("inf"), float("-inf")):
    try: usage("u", score=score)
    except ValueError: pass
    else: raise AssertionError("non-finite score accepted")
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-integration-usage-filter",
        ("pico/knowledge_evolution/store.py", "tests/*"),
        ("tests/test_knowledge_usage.py",),
        _USAGE_PRELUDE
        + r"""
with TemporaryDirectory() as root:
    store=KnowledgeRecordStore(Path(root)); store.write_usage(usage("u1",candidate="c1",mode=KnowledgeUsageMode.INJECTED)); store.write_usage(usage("u2",candidate="c2",mode=KnowledgeUsageMode.REFERENCED))
    assert tuple(x.usage_id for x in store.list_usages(candidate_id="c2")) == ("u2",)
    assert tuple(x.usage_id for x in store.list_usages(usage_mode=KnowledgeUsageMode.INJECTED)) == ("u1",)
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-integration-turn-retrieval-count",
        ("pico/knowledge_evolution/retrieval.py", "pico/knowledge_evolution/__init__.py", "tests/*"),
        ("tests/test_knowledge_retrieval.py",),
        _RETRIEVAL_PRELUDE
        + r"""
from pico.knowledge_evolution import count_turn_retrievals
with TemporaryDirectory() as root:
    store=KnowledgeRecordStore(Path(root)); store.write_retrieval(receipt("r1","t1")); store.write_retrieval(receipt("r2","t1")); store.write_retrieval(receipt("r3","t2"))
    assert count_turn_retrievals(store,"t1") == 2 and count_turn_retrievals(store,"missing") == 0
""",
    ),
    OfficialVerifierSpec(
        "p3r4-v-integration-suite-registration",
        ("scripts/testing/test_matrix.json", "tests/*"),
        ("tests/test_test_runner.py",),
        r"""
import json
from pathlib import Path
data=json.loads(Path("scripts/testing/test_matrix.json").read_text(encoding="utf-8"))
targets=data["suites"]["p3_official_contract"]["targets"]
assert targets == ["tests/test_knowledge_retrieval.py","tests/test_knowledge_usage.py"]
assert len(targets) == len(set(targets))
""",
    ),
)

VERIFIERS = {item.verifier_id: item for item in (*_SPECS, *_OFFICIAL_SPECS)}


def changed_path_findings(
    spec: HiddenVerifierSpec,
    changed: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    findings: list[str] = []
    for pattern in spec.required_changed_paths:
        if not any(fnmatch.fnmatch(path, pattern) for path in changed):
            findings.append(
                "regression_test_missing" if pattern.startswith("tests/") else "missing_required_path_class"
            )
    if spec.required_changed_path_alternatives and not any(
        all(any(fnmatch.fnmatch(path, pattern) for path in changed) for pattern in alternative)
        for alternative in spec.required_changed_path_alternatives
    ):
        findings.append("missing_required_path_class")
    if any(path.startswith((".p3r/", ".pico/")) for path in changed):
        findings.append("prohibited_change_detected")
    return tuple(dict.fromkeys(findings))


def has_two_source_router_proof(source: str) -> bool:
    """Recognize a deterministic integration proof without depending on test names."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        sources = node.args[0]
        if name == "SkillForgeRouter" and isinstance(sources, (ast.List, ast.Tuple)) and len(sources.elts) >= 2:
            return True
    return False


def integration_proof_findings(
    spec: HiddenVerifierSpec,
    workspace: Path,
    changed: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Require actual two-source proof when the integration outcome is proof-only."""

    if spec.verifier_id != "p3r-v-integration-skill":
        return ()
    production_changed = any(fnmatch.fnmatch(path, "pico/memory_engine/skill_forge/*") for path in changed)
    proof_path = "tests/test_knowledge_runtime_integration.py"
    if production_changed or proof_path not in changed:
        return ()
    try:
        source = (workspace / proof_path).read_text(encoding="utf-8")
    except OSError:
        return ("regression_test_missing",)
    return () if has_two_source_router_proof(source) else ("regression_test_missing",)


_DEBUG_SEMANTIC_PROBE = r"""
import json
from pico.knowledge_evolution.canonicalize import KnowledgeProposal, ProposalRejectionReason, validate_and_canonicalize

def proposal(applicability):
    return KnowledgeProposal(
        candidate_type="experience",
        content_class="recovery",
        title="Guard probe",
        reusable_content="Validate deterministic guard behavior.",
        applicability=applicability,
    )

recognized = validate_and_canonicalize(proposal((
    "tool:read_file", "binary:git", "file:pyproject.toml", "dependency:pytest"
)))
reordered = validate_and_canonicalize(proposal((
    "dependency:pytest", "file:pyproject.toml", "binary:git", "tool:read_file"
)))
opaque = validate_and_canonicalize(proposal(("unsupported:value",)))
duplicate = validate_and_canonicalize(proposal(("tool:read_file", "tool:read_file")))
malformed = validate_and_canonicalize(proposal((object(),)))

recognized_keys = tuple(key for key, _ in recognized.proposal.applicability_fingerprints) if recognized.proposal else ()
reordered_fingerprints = reordered.proposal.applicability_fingerprints if reordered.proposal else ()
opaque_keys = tuple(key for key, _ in opaque.proposal.applicability_fingerprints) if opaque.proposal else ()
duplicate_fingerprints = duplicate.proposal.applicability_fingerprints if duplicate.proposal else ()
print(json.dumps({
    "recognized": recognized_keys == (
        "guard:binary:git",
        "guard:dependency:pytest",
        "guard:file:pyproject.toml",
        "guard:tool:read_file",
    ),
    "deterministic": recognized.proposal is not None and recognized.proposal.applicability_fingerprints == reordered_fingerprints,
    "opaque_fail_closed": opaque_keys == ("guard:0",),
    "deduplicated": len(duplicate_fingerprints) == 1,
    "malformed_rejected": malformed.proposal is None and malformed.reason == ProposalRejectionReason.MALFORMED_PROPOSAL,
}, sort_keys=True))
"""


def debug_semantic_findings(
    spec: HiddenVerifierSpec,
    workspace: Path,
    *,
    python_executable: str,
) -> tuple[str, ...]:
    """Validate guard behavior without coupling to source layout or helper names."""

    if spec.verifier_id != "p3r-v-debug-guard":
        return ()
    try:
        completed = subprocess.run(
            [python_executable, "-c", _DEBUG_SEMANTIC_PROBE],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ("verifier_host_failure",)
    try:
        results = json.loads(completed.stdout.strip()) if completed.returncode == 0 else {}
    except json.JSONDecodeError:
        results = {}
    required = (
        "recognized",
        "deterministic",
        "opaque_fail_closed",
        "deduplicated",
        "malformed_rejected",
    )
    return () if all(results.get(item) is True for item in required) else ("guard_semantics_failed",)


def official_semantic_findings(
    spec: HiddenVerifierSpec | OfficialVerifierSpec,
    workspace: Path,
    *,
    python_executable: str,
) -> tuple[str, ...]:
    if not isinstance(spec, OfficialVerifierSpec):
        return ()
    try:
        completed = subprocess.run(
            [python_executable, "-c", spec.semantic_probe],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ("verifier_host_failure",)
    return () if completed.returncode == 0 else ("semantic_invariant_failed",)


def verify_workspace(
    verifier_id: str,
    workspace: Path,
    *,
    python_executable: str,
) -> HiddenVerifierResult:
    """Verify only after terminal state; never feed findings back to the Agent."""

    spec = VERIFIERS[verifier_id]
    try:
        changed = _git(workspace, "diff", "--name-only", "HEAD").splitlines()
        changed.extend(_git(workspace, "ls-files", "--others", "--exclude-standard").splitlines())
    except (OSError, RuntimeError):
        return HiddenVerifierResult(
            False,
            ("verifier_host_failure",),
            spec.verifier_id,
            spec.digest,
            infrastructure_failure=True,
        )
    findings = list(changed_path_findings(spec, changed))
    findings.extend(integration_proof_findings(spec, workspace, changed))
    findings.extend(debug_semantic_findings(spec, workspace, python_executable=python_executable))
    findings.extend(official_semantic_findings(spec, workspace, python_executable=python_executable))
    for relative, marker in spec.required_source_markers:
        path = workspace / relative
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            findings.append("expected_symbol_missing")
            continue
        if marker not in text:
            findings.append("expected_symbol_missing")
    try:
        completed = subprocess.run(
            [python_executable, "-m", "pytest", "-q", *spec.targeted_tests],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return HiddenVerifierResult(
            False,
            ("verifier_host_failure",),
            spec.verifier_id,
            spec.digest,
            infrastructure_failure=True,
        )
    if completed.returncode != 0:
        findings.append("targeted_test_failed")
    normalized = tuple(dict.fromkeys(findings))
    return HiddenVerifierResult(
        not normalized,
        normalized,
        spec.verifier_id,
        spec.digest,
        infrastructure_failure="verifier_host_failure" in normalized,
    )


def _git(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],  # noqa: S607 -- repository-native executable
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


__all__ = [
    "VERIFIERS",
    "HiddenVerifierResult",
    "HiddenVerifierSpec",
    "OfficialVerifierSpec",
    "changed_path_findings",
    "debug_semantic_findings",
    "has_two_source_router_proof",
    "integration_proof_findings",
    "official_semantic_findings",
    "verify_workspace",
]
