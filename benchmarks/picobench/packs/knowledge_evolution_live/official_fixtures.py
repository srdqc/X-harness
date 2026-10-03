"""Sealed mechanical reference fixtures for the frozen P3R.4 official suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from benchmarks.picobench.canonical import canonical_digest

from .tasks import OFFICIAL_TASKS_V1, OFFICIAL_TASKS_V2

OFFICIAL_SUITE_VERSION_V1 = "p3r4-held-out-v1"
OFFICIAL_SUITE_VERSION = "p3r4-held-out-v2"
OFFICIAL_REFERENCE_BASE = "f5ac937b091905786a88cfc9abfbe0c2fa9bb22d"


def _replace(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if old not in source:
        raise RuntimeError(f"reference fixture anchor missing: {path.name}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _touch_test(root: Path, relative: str, task_id: str) -> None:
    path = root / relative
    path.write_text(
        path.read_text(encoding="utf-8") + f"\n\n# Sealed mechanical reference exercised by {task_id}.\n",
        encoding="utf-8",
    )


def _store_method(root: Path, anchor: str, method: str) -> None:
    path = root / "pico/knowledge_evolution/store.py"
    _replace(path, anchor, method + "\n" + anchor)


def _nav_retrieval(root: Path) -> None:
    _store_method(
        root,
        "    def write_usage(self, receipt: Any) -> ImmutableWriteStatus:\n",
        """    def list_retrievals(self, *, turn_id: str | None = None) -> tuple[Any, ...]:
        if not self.retrievals.exists():
            return ()
        from .retrieval import KnowledgeRetrievalReceipt
        values = []
        for path in sorted(self.retrievals.glob("*.json"), key=lambda item: item.name):
            value = _load(path, KnowledgeRetrievalReceipt.from_dict)
            if value.retrieval_id != path.stem:
                raise KnowledgeStoreError("retrieval receipt path binding mismatch")
            if turn_id is None or value.turn_id == turn_id:
                values.append(value)
        return tuple(values)
""",
    )


def _nav_outcome(root: Path) -> None:
    _store_method(
        root,
        "    def load_or_create_local_binding(\n",
        """    def list_outcome_associations(self, *, turn_id: str | None = None) -> tuple[Any, ...]:
        if not self.outcome_associations.exists():
            return ()
        from .usage import KnowledgeUsageOutcomeAssociation
        values = []
        for path in sorted(self.outcome_associations.glob("*.json"), key=lambda item: item.name):
            value = _load(path, KnowledgeUsageOutcomeAssociation.from_dict)
            if value.association_id != path.stem:
                raise KnowledgeStoreError("outcome association path binding mismatch")
            if turn_id is None or value.turn_id == turn_id:
                values.append(value)
        return tuple(values)
""",
    )


def _nav_applicability(root: Path) -> None:
    _store_method(
        root,
        "    def write_retrieval(self, receipt: Any) -> ImmutableWriteStatus:\n",
        """    def list_applicability_results(self, *, candidate_id: str | None = None, status: Any | None = None) -> tuple[Any, ...]:
        if not self.applicability_results.exists():
            return ()
        from .applicability import KnowledgeApplicabilityResult
        values = []
        for path in sorted(self.applicability_results.glob("*.json"), key=lambda item: item.name):
            value = _load(path, KnowledgeApplicabilityResult.from_dict)
            if value.applicability_id != path.stem:
                raise KnowledgeStoreError("applicability result path binding mismatch")
            if candidate_id is not None and value.candidate_id != candidate_id:
                continue
            if status is not None and value.status != status:
                continue
            values.append(value)
        return tuple(values)
""",
    )


def _nav_applicability_alternate(root: Path) -> None:
    _store_method(
        root,
        "    def write_retrieval(self, receipt: Any) -> ImmutableWriteStatus:\n",
        """    def list_applicability_results(self, *, candidate_id: str | None = None, status: Any | None = None) -> tuple[Any, ...]:
        from .applicability import KnowledgeApplicabilityResult
        def matches(value: Any) -> bool:
            return ((candidate_id is None or value.candidate_id == candidate_id) and (status is None or value.status == status))
        if not self.applicability_results.exists():
            return ()
        results = []
        for path in sorted(self.applicability_results.glob("*.json")):
            value = _load(path, KnowledgeApplicabilityResult.from_dict)
            if path.stem != value.applicability_id:
                raise KnowledgeStoreError("applicability result path binding mismatch")
            if matches(value):
                results.append(value)
        return tuple(results)
""",
    )


def _nav_applicability_missing(root: Path) -> None:
    _touch_test(root, "tests/test_knowledge_applicability.py", "p3r4-nav-03-missing")


def _nav_applicability_broken_candidate(root: Path) -> None:
    _nav_applicability(root)
    path = root / "pico/knowledge_evolution/store.py"
    _replace(
        path,
        "            if candidate_id is not None and value.candidate_id != candidate_id:\n",
        "            if False and candidate_id is not None and value.candidate_id != candidate_id:\n",
    )


def _nav_applicability_broken_status(root: Path) -> None:
    _nav_applicability(root)
    path = root / "pico/knowledge_evolution/store.py"
    _replace(
        path,
        "            if status is not None and value.status != status:\n",
        "            if False and status is not None and value.status != status:\n",
    )


def _nav_applicability_noncomposable(root: Path) -> None:
    _nav_applicability(root)
    path = root / "pico/knowledge_evolution/store.py"
    _replace(
        path,
        "        from .applicability import KnowledgeApplicabilityResult\n        values = []\n",
        "        from .applicability import KnowledgeApplicabilityResult\n        if candidate_id is not None and status is not None:\n            status = None\n        values = []\n",
    )


def _nav_applicability_unordered(root: Path) -> None:
    _nav_applicability(root)
    path = root / "pico/knowledge_evolution/store.py"
    _replace(
        path,
        '        for path in sorted(self.applicability_results.glob("*.json"), key=lambda item: item.name):\n',
        '        for path in sorted(self.applicability_results.glob("*.json"), key=lambda item: item.name, reverse=True):\n',
    )


def _nav_applicability_broken_identity(root: Path) -> None:
    _nav_applicability(root)
    path = root / "pico/knowledge_evolution/store.py"
    _replace(
        path,
        '            if value.applicability_id != path.stem:\n                raise KnowledgeStoreError("applicability result path binding mismatch")\n',
        "",
    )


def _retrieval_summary(root: Path) -> None:
    path = root / "pico/knowledge_evolution/retrieval.py"
    _replace(
        path,
        "    def _payload(self) -> dict[str, Any]:\n",
        """    @property
    def selection_summary(self) -> dict[str, int]:
        return {"ranked": len(self.ranked_candidate_ids), "selected": len(self.selected_candidate_ids), "suppressed": len(self.suppressed)}

    def _payload(self) -> dict[str, Any]:
""",
    )


def _usage_visibility(root: Path) -> None:
    path = root / "pico/knowledge_evolution/usage.py"
    _replace(
        path,
        "    def _payload(self) -> dict[str, Any]:\n",
        """    @property
    def is_model_visible(self) -> bool:
        return self.usage_mode is not KnowledgeUsageMode.RETRIEVED

    def _payload(self) -> dict[str, Any]:
""",
    )


def _outcome_counts(root: Path) -> None:
    path = root / "pico/knowledge_evolution/usage.py"
    anchor = "    def _payload(self) -> dict[str, Any]:\n"
    first = path.read_text(encoding="utf-8").find(anchor)
    second = path.read_text(encoding="utf-8").find(anchor, first + 1)
    source = path.read_text(encoding="utf-8")
    if second < 0:
        raise RuntimeError("outcome reference anchor missing")
    addition = """    @property
    def usage_count(self) -> int:
        return len(self.usage_ids)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

"""
    path.write_text(source[:second] + addition + source[second:], encoding="utf-8")


def _duplicate_usage(root: Path) -> None:
    path = root / "pico/knowledge_evolution/usage.py"
    _replace(
        path,
        "        if not self.usage_ids or len(self.usage_ids) != len(self.usage_digests):\n",
        '        if len(self.usage_ids) != len(set(self.usage_ids)):\n            raise ValueError("usage IDs must be unique")\n        if not self.usage_ids or len(self.usage_ids) != len(self.usage_digests):\n',
    )


def _retrieval_membership(root: Path) -> None:
    path = root / "pico/knowledge_evolution/retrieval.py"
    _replace(
        path,
        "        if self.applicable_count != len(self.ranked_candidate_ids):\n",
        '        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):\n            raise ValueError("selected candidate IDs must be unique")\n        if not set(self.selected_candidate_ids).issubset(self.ranked_candidate_ids):\n            raise ValueError("selected candidates must be ranked")\n        if self.applicable_count != len(self.ranked_candidate_ids):\n',
    )


def _finite_score(root: Path) -> None:
    path = root / "pico/knowledge_evolution/usage.py"
    _replace(path, "from enum import Enum\n", "from enum import Enum\nimport math\n")
    _replace(
        path,
        "        if self.retrieval_rank < 1:\n",
        '        if not math.isfinite(self.retrieval_score):\n            raise ValueError("retrieval_score must be finite")\n        if self.retrieval_rank < 1:\n',
    )


def _usage_filter(root: Path) -> None:
    path = root / "pico/knowledge_evolution/store.py"
    _replace(
        path,
        "    def list_usages(self, *, turn_id: str | None = None) -> tuple[Any, ...]:\n",
        "    def list_usages(self, *, turn_id: str | None = None, candidate_id: str | None = None, usage_mode: Any | None = None) -> tuple[Any, ...]:\n",
    )
    _replace(
        path,
        "            if turn_id is None or value.turn_id == turn_id:\n                values.append(value)\n        return tuple(values)\n\n    def write_outcome_association",
        "            if turn_id is not None and value.turn_id != turn_id:\n                continue\n            if candidate_id is not None and value.candidate_id != candidate_id:\n                continue\n            if usage_mode is not None and value.usage_mode != usage_mode:\n                continue\n            values.append(value)\n        return tuple(values)\n\n    def write_outcome_association",
    )


def _turn_retrieval_count(root: Path) -> None:
    _nav_retrieval(root)
    retrieval = root / "pico/knowledge_evolution/retrieval.py"
    retrieval.write_text(
        retrieval.read_text(encoding="utf-8")
        + """

def count_turn_retrievals(store: KnowledgeRecordStore, turn_id: str) -> int:
    return len(store.list_retrievals(turn_id=turn_id))
""",
        encoding="utf-8",
    )
    init = root / "pico/knowledge_evolution/__init__.py"
    _replace(init, "    KnowledgeRetrievalReceipt,\n", "    KnowledgeRetrievalReceipt,\n    count_turn_retrievals,\n")
    _replace(
        init, '    "build_extraction_context",\n', '    "build_extraction_context",\n    "count_turn_retrievals",\n'
    )


def _suite_registration(root: Path) -> None:
    path = root / "scripts/testing/test_matrix.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["suites"]["p3_official_contract"] = {
        "targets": ["tests/test_knowledge_retrieval.py", "tests/test_knowledge_usage.py"]
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


_APPLIERS: dict[str, tuple[Callable[[Path], None], str]] = {
    "p3r4-nav-01": (_nav_retrieval, "tests/test_knowledge_retrieval.py"),
    "p3r4-nav-02": (_nav_outcome, "tests/test_knowledge_usage.py"),
    "p3r4-nav-03": (_nav_applicability, "tests/test_knowledge_applicability.py"),
    "p3r4-impl-01": (_retrieval_summary, "tests/test_knowledge_retrieval.py"),
    "p3r4-impl-02": (_usage_visibility, "tests/test_knowledge_usage.py"),
    "p3r4-impl-03": (_outcome_counts, "tests/test_knowledge_usage.py"),
    "p3r4-debug-01": (_duplicate_usage, "tests/test_knowledge_usage.py"),
    "p3r4-debug-02": (_retrieval_membership, "tests/test_knowledge_retrieval.py"),
    "p3r4-debug-03": (_finite_score, "tests/test_knowledge_usage.py"),
    "p3r4-int-01": (_usage_filter, "tests/test_knowledge_usage.py"),
    "p3r4-int-02": (_turn_retrieval_count, "tests/test_knowledge_retrieval.py"),
    "p3r4-int-03": (_suite_registration, "tests/test_test_runner.py"),
}


def apply_official_reference(
    task_id: str, workspace: Path, *, suite_version: str = OFFICIAL_SUITE_VERSION
) -> None:
    if suite_version not in {OFFICIAL_SUITE_VERSION_V1, OFFICIAL_SUITE_VERSION}:
        raise ValueError(f"unsupported official suite version: {suite_version}")
    apply, test = _APPLIERS[task_id]
    apply(workspace)
    _touch_test(workspace, test, task_id)


def official_reference_digest(
    task_id: str, *, suite_version: str = OFFICIAL_SUITE_VERSION
) -> str:
    tasks = OFFICIAL_TASKS_V1 if suite_version == OFFICIAL_SUITE_VERSION_V1 else OFFICIAL_TASKS_V2
    task = next(item for item in tasks if item.task_id == task_id)
    return canonical_digest(
        {
            "suite": suite_version,
            "task_id": task_id,
            "verifier": task.verifier_digest,
            "fixture_version": 1 if suite_version == OFFICIAL_SUITE_VERSION_V1 else 2,
        }
    )


def official_reference_digests(
    *, suite_version: str = OFFICIAL_SUITE_VERSION
) -> tuple[tuple[str, str], ...]:
    tasks = OFFICIAL_TASKS_V1 if suite_version == OFFICIAL_SUITE_VERSION_V1 else OFFICIAL_TASKS_V2
    return tuple(
        (task.task_id, official_reference_digest(task.task_id, suite_version=suite_version))
        for task in tasks
    )


NAV03_SEMANTIC_FIXTURES: tuple[tuple[str, Callable[[Path], None], bool], ...] = (
    ("canonical", _nav_applicability, True),
    ("alternate-helper", _nav_applicability_alternate, True),
    ("missing-api", _nav_applicability_missing, False),
    ("broken-candidate", _nav_applicability_broken_candidate, False),
    ("broken-status", _nav_applicability_broken_status, False),
    ("noncomposable", _nav_applicability_noncomposable, False),
    ("unordered", _nav_applicability_unordered, False),
    ("broken-identity", _nav_applicability_broken_identity, False),
)


def apply_nav03_semantic_fixture(fixture_id: str, workspace: Path) -> None:
    apply, _expected = next(
        (apply, expected)
        for name, apply, expected in NAV03_SEMANTIC_FIXTURES
        if name == fixture_id
    )
    apply(workspace)
    if fixture_id != "missing-api":
        _touch_test(workspace, "tests/test_knowledge_applicability.py", fixture_id)


__all__ = [
    "OFFICIAL_REFERENCE_BASE",
    "OFFICIAL_SUITE_VERSION",
    "OFFICIAL_SUITE_VERSION_V1",
    "NAV03_SEMANTIC_FIXTURES",
    "apply_nav03_semantic_fixture",
    "apply_official_reference",
    "official_reference_digest",
    "official_reference_digests",
]
