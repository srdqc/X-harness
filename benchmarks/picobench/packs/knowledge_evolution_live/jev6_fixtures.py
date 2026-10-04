"""Sealed mechanical references for the JEV.6 held-out suite."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from benchmarks.picobench.canonical import canonical_digest

from .jev6_suite import SUITE_VERSION, TASKS


def _replace(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if old not in source:
        raise RuntimeError(f"JEV.6 reference anchor missing: {path}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _touch_test(root: Path, relative: str, task_id: str) -> None:
    path = root / relative
    path.write_text(
        path.read_text(encoding="utf-8")
        + f"\n\n# Sealed JEV.6 mechanical reference: {task_id}.\n",
        encoding="utf-8",
    )


_READ_EVENTS = '''    def read_events(
        self,
        *,
        turn_id: str | None = None,
        event_type: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        path = self.logs_dir / _KIND_FILES["events"]
        if not path.exists():
            return ()
        values: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("malformed trace event JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError("trace event record must be an object")
            if turn_id is not None and value.get("turn_id") != turn_id:
                continue
            if event_type is not None and value.get("event_type") != event_type:
                continue
            values.append(value)
        return tuple(values)

'''


def _add_read_events(root: Path) -> None:
    path = root / "pico/tracing/store.py"
    _replace(path, "    def persist_artifact(\n", _READ_EVENTS + "    def persist_artifact(\n")


def _trace_event_reader(root: Path) -> None:
    _add_read_events(root)
    _touch_test(root, "tests/test_tracing_api.py", "jev6-nav-01-trace-event-reader")


def _skill_variants(root: Path) -> None:
    path = root / "pico/memory_engine/skill_local/registry.py"
    _replace(
        path,
        "    def get(self, name: str, source: str | None = None) -> SkillMeta | None:\n",
        '''    def list_variants(self, name: str) -> tuple[SkillMeta, ...]:
        if self._by_name is None:
            self.list_all()
        return tuple(sorted(
            (meta for meta in (self._metas_cache or ()) if meta.name == name),
            key=lambda meta: (meta.source, meta.id),
        ))

    def get(self, name: str, source: str | None = None) -> SkillMeta | None:
''',
    )
    _touch_test(root, "tests/test_skill_forge_local_pool.py", "jev6-nav-02-skill-variants")


def _safe_segment_bound(root: Path) -> None:
    path = root / "pico/tracing/store.py"
    _replace(
        path,
        'def safe_segment(value: Any, fallback: str = "unknown") -> str:\n    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str("" if value is None else value).strip())\n    normalized = normalized.strip("-")\n    return (normalized or fallback)[:80]\n',
        '''def safe_segment(value: Any, fallback: str = "unknown", *, max_length: int = 80) -> str:
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str("" if value is None else value).strip())
    normalized = normalized.strip("-")
    return (normalized or fallback)[:max_length]
''',
    )
    _touch_test(root, "tests/test_tracing_api.py", "jev6-impl-01-safe-segment-bound")


def _delivery_result_flags(root: Path) -> None:
    path = root / "pico/spine/delivery.py"
    _replace(
        path,
        "\n\n@dataclass(frozen=True)\nclass Capabilities:\n",
        '''

    @property
    def succeeded(self) -> bool:
        return self.outcome == "delivered"

    @property
    def retry_exhausted(self) -> bool:
        return self.outcome == "dropped" and self.attempts > 1


@dataclass(frozen=True)
class Capabilities:
''',
    )
    _touch_test(root, "tests/test_spine_delivery.py", "jev6-impl-02-delivery-result-flags")


def _bm25_parameters(root: Path) -> None:
    path = root / "pico/utils/bm25.py"
    _replace(
        path,
        "    ) -> None:\n        self.k1 = k1\n        self.b = b\n",
        '''    ) -> None:
        if isinstance(k1, bool) or not isinstance(k1, (int, float)) or not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and greater than zero")
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and within [0, 1]")
        self.k1 = k1
        self.b = b
''',
    )
    _touch_test(root, "tests/test_bm25.py", "jev6-debug-01-bm25-parameters")


def _decision_receipt_identity(root: Path) -> None:
    path = root / "pico/decision_plane/evidence.py"
    _replace(path, "import math\n", "import math\nimport re\n")
    _replace(
        path,
        "        if self.candidate_count < 0 or not math.isfinite(self.latency_ms) or self.latency_ms < 0:\n",
        '''        digest_fields = (
            self.candidate_set_digest,
            self.request_digest,
            self.baseline_result_digest,
            self.adapter_result_digest,
            self.final_result_digest,
        )
        if any(value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digest_fields):
            raise ValueError("decision receipt digest is invalid")
        confidence_ids = tuple(candidate_id for candidate_id, _ in self.confidences)
        if len(confidence_ids) != len(set(confidence_ids)):
            raise ValueError("decision confidence candidate IDs must be unique")
        if self.candidate_count < 0 or not math.isfinite(self.latency_ms) or self.latency_ms < 0:
''',
    )
    _touch_test(root, "tests/test_execution_receipts.py", "jev6-debug-02-decision-receipt-identity")


def _public_turn_events(root: Path) -> None:
    _add_read_events(root)
    store = root / "pico/tracing/store.py"
    store.write_text(
        store.read_text(encoding="utf-8")
        + '''

def read_turn_events(store: TraceStore, turn_id: str) -> tuple[dict[str, Any], ...]:
    return store.read_events(turn_id=turn_id)
''',
        encoding="utf-8",
    )
    init = root / "pico/tracing/__init__.py"
    _replace(
        init,
        "from . import config, evidence, replay, trace, verifier\n",
        "from . import config, evidence, replay, trace, verifier\nfrom .store import TraceStore, read_turn_events\n",
    )
    _replace(
        init,
        '__all__ = ["enabled", "evidence", "replay", "trace", "verifier"]\n',
        '__all__ = ["TraceStore", "enabled", "evidence", "read_turn_events", "replay", "trace", "verifier"]\n',
    )
    _touch_test(root, "tests/test_tracing_api.py", "jev6-int-01-public-turn-events")


def _resolve_available_skill(root: Path) -> None:
    path = root / "pico/memory_engine/skill_local/registry.py"
    _replace(
        path,
        "    def get_body(self, name: str, source: str | None = None) -> str | None:\n",
        '''    def resolve_available(self, name: str, source: str | None = None) -> SkillMeta | None:
        meta = self.get(name, source=source)
        if meta is None or not self.check_available(name, source=source):
            return None
        return meta

    def get_body(self, name: str, source: str | None = None) -> str | None:
''',
    )
    _touch_test(root, "tests/test_skill_forge_local_pool.py", "jev6-int-02-resolve-available-skill")


_APPLIERS: dict[str, tuple[Callable[[Path], None], str]] = {
    "jev6-nav-01-trace-event-reader": (_trace_event_reader, "tests/test_tracing_api.py"),
    "jev6-nav-02-skill-variants": (_skill_variants, "tests/test_skill_forge_local_pool.py"),
    "jev6-impl-01-safe-segment-bound": (_safe_segment_bound, "tests/test_tracing_api.py"),
    "jev6-impl-02-delivery-result-flags": (_delivery_result_flags, "tests/test_spine_delivery.py"),
    "jev6-debug-01-bm25-parameters": (_bm25_parameters, "tests/test_bm25.py"),
    "jev6-debug-02-decision-receipt-identity": (_decision_receipt_identity, "tests/test_execution_receipts.py"),
    "jev6-int-01-public-turn-events": (_public_turn_events, "tests/test_tracing_api.py"),
    "jev6-int-02-resolve-available-skill": (_resolve_available_skill, "tests/test_skill_forge_local_pool.py"),
}


def apply_reference(task_id: str, workspace: Path) -> None:
    apply, _test = _APPLIERS[task_id]
    apply(workspace)


def reference_digest(task_id: str) -> str:
    task = next(task for task in TASKS if task.task_id == task_id)
    return canonical_digest(
        {
            "suite_version": SUITE_VERSION,
            "task_id": task_id,
            "prompt_digest": task.prompt_digest,
            "verifier_digest": task.verifier_digest,
            "fixture_version": 1,
        }
    )


def reference_digests() -> tuple[tuple[str, str], ...]:
    return tuple((task.task_id, reference_digest(task.task_id)) for task in TASKS)


__all__ = ["apply_reference", "reference_digest", "reference_digests"]
