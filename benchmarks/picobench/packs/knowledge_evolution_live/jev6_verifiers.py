"""Independent post-Turn semantic verifiers for the frozen JEV.6 tasks."""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest

from .jev6_suite import TASKS, verifier_digest


@dataclass(frozen=True)
class VerifierSpec:
    verifier_id: str
    production_paths: tuple[str, ...]
    test_path: str
    semantic_probe: str
    run_after_terminal: bool = True
    version: int = 1

    @property
    def digest(self) -> str:
        return verifier_digest(self.verifier_id)


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    findings: tuple[str, ...]
    verifier_id: str
    verifier_digest: str


_TRACE_READER_PROBE = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from pico.tracing.store import TraceStore
with TemporaryDirectory() as root:
    store=TraceStore(Path(root))
    assert store.read_events() == ()
    store.append_event({"turn_id":"t1","event_type":"A","order":1})
    store.append_event({"turn_id":"t2","event_type":"A","order":2})
    store.append_event({"turn_id":"t1","event_type":"B","order":3})
    assert [x["order"] for x in store.read_events()] == [1,2,3]
    assert [x["order"] for x in store.read_events(turn_id="t1", event_type="B")] == [3]
    path=Path(root)/"logs"/"audit-events.log"; path.write_text(path.read_text()+"not-json\n",encoding="utf-8")
    try: store.read_events()
    except ValueError: pass
    else: raise AssertionError("malformed JSONL did not fail closed")
'''

_SKILL_VARIANTS_PROBE = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from pico.memory_engine.skill_local.registry import SkillRegistry
with TemporaryDirectory() as root:
    root=Path(root); workspace=root/"workspace"; builtin=root/"builtin"
    for base, source in ((workspace/"skills"/"demo","workspace"),(builtin/"demo","builtin")):
        base.mkdir(parents=True); (base/"SKILL.md").write_text("---\nname: shared\ndescription: demo\n---\nbody\n",encoding="utf-8")
    registry=SkillRegistry(workspace,builtin_skills_dir=builtin)
    registry.list_all(); cache=registry._metas_cache
    values=registry.list_variants("shared")
    assert [(x.source,x.name) for x in values] == [("builtin","shared"),("workspace","shared")]
    assert registry._metas_cache is cache
    assert registry.list_variants("missing") == ()
'''

_SAFE_SEGMENT_PROBE = r'''
from pico.tracing.store import safe_segment
assert safe_segment("a b") == "a-b"
assert len(safe_segment("x"*100)) == 80
assert safe_segment("x"*20,max_length=7) == "x"*7
assert safe_segment("***",fallback="fallback",max_length=4) == "fall"
for value in (0,-1,True,1.5):
    try: safe_segment("x",max_length=value)
    except ValueError: pass
    else: raise AssertionError("invalid max_length accepted")
'''

_DELIVERY_FLAGS_PROBE = r'''
from pico.spine.delivery import DeliveryResult
def item(outcome,attempts): return DeliveryResult(None,None,"cli","Text",outcome,attempts)
assert item("delivered",1).succeeded is True
assert item("dropped",3).succeeded is False
assert item("dropped",2).retry_exhausted is True
assert item("dropped",1).retry_exhausted is False
assert item("delivered",3).retry_exhausted is False
assert tuple(item("delivered",1).__dataclass_fields__) == ("conversation_id","turn_id","channel","event","outcome","attempts","error")
'''

_BM25_PROBE = r'''
import math
from pico.utils.bm25 import BM25Okapi
assert BM25Okapi([["alpha"]]).get_scores(["alpha"])[0] > 0
for k1,b in ((0,0.75),(-1,0.75),(True,0.75),(math.nan,0.75),(math.inf,0.75),(1.5,-0.1),(1.5,1.1),(1.5,True),(1.5,math.nan),(1.5,math.inf)):
    try: BM25Okapi([["alpha"]],k1=k1,b=b)
    except ValueError: pass
    else: raise AssertionError("invalid BM25 parameter accepted")
'''

_DECISION_RECEIPT_PROBE = r'''
from pico.decision_plane.evidence import DecisionReceipt
def item(**changes):
    values=dict(turn_id="t",decision_id="d",decision_type="rank",adapter_kind="fake",candidate_count=1,candidate_set_digest="a"*64,request_digest="b"*64,baseline_result_digest="c"*64,adapter_result_digest="d"*64,final_result_digest="e"*64,adapter_outcome="success",fallback_used=False,fallback_reason=None,latency_ms=1.0,cost_available=False,cost_amount=None,cost_unit=None,confidences=(("c1",0.5),),final_ranking_source="adapter")
    values.update(changes); return DecisionReceipt(**values)
valid=item(); assert valid.metadata()["request_digest"] == "b"*64
for changes in ({"request_digest":"bad"},{"final_result_digest":"A"*64},{"adapter_result_digest":"g"*64},{"confidences":(("c1",0.2),("c1",0.8))}):
    try: item(**changes)
    except ValueError: pass
    else: raise AssertionError("invalid decision identity accepted")
'''

_PUBLIC_TURN_EVENTS_PROBE = _TRACE_READER_PROBE + r'''
from pico.tracing import TraceStore as PublicTraceStore, read_turn_events
with TemporaryDirectory() as root:
    store=PublicTraceStore(Path(root)); store.append_event({"turn_id":"t","event_type":"A","order":1}); store.append_event({"turn_id":"u","event_type":"A","order":2})
    assert [x["order"] for x in read_turn_events(store,"t")] == [1]
    assert read_turn_events(store,"missing") == ()
import pico.tracing as tracing
assert "read_turn_events" in tracing.__all__
'''

_RESOLVE_AVAILABLE_PROBE = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from pico.memory_engine.skill_local.registry import SkillRegistry
with TemporaryDirectory() as root:
    root=Path(root); workspace=root/"workspace"; builtin=root/"builtin"
    ready=workspace/"skills"/"ready"; blocked=workspace/"skills"/"blocked"
    ready.mkdir(parents=True); blocked.mkdir(parents=True); builtin.mkdir()
    ready.joinpath("SKILL.md").write_text("---\nname: ready\ndescription: ready\n---\nbody\n",encoding="utf-8")
    blocked.joinpath("SKILL.md").write_text('---\nname: blocked\ndescription: blocked\nmetadata: {"pico":{"requires":{"env":["JEV6_MISSING_ENV"]}}}\n---\nbody\n',encoding="utf-8")
    registry=SkillRegistry(workspace,builtin_skills_dir=builtin)
    assert registry.resolve_available("ready",source="workspace").name == "ready"
    assert registry.resolve_available("blocked",source="workspace") is None
    assert registry.resolve_available("missing") is None
'''


SPECS = (
    VerifierSpec("jev6-v-trace-event-reader", ("pico/tracing/store.py",), "tests/test_tracing_api.py", _TRACE_READER_PROBE),
    VerifierSpec("jev6-v-skill-variants", ("pico/memory_engine/skill_local/registry.py",), "tests/test_skill_forge_local_pool.py", _SKILL_VARIANTS_PROBE),
    VerifierSpec("jev6-v-safe-segment-bound", ("pico/tracing/store.py",), "tests/test_tracing_api.py", _SAFE_SEGMENT_PROBE),
    VerifierSpec("jev6-v-delivery-result-flags", ("pico/spine/delivery.py",), "tests/test_spine_delivery.py", _DELIVERY_FLAGS_PROBE),
    VerifierSpec("jev6-v-bm25-parameters", ("pico/utils/bm25.py",), "tests/test_bm25.py", _BM25_PROBE),
    VerifierSpec("jev6-v-decision-receipt-identity", ("pico/decision_plane/evidence.py",), "tests/test_execution_receipts.py", _DECISION_RECEIPT_PROBE),
    VerifierSpec("jev6-v-public-turn-events", ("pico/tracing/store.py", "pico/tracing/__init__.py"), "tests/test_tracing_api.py", _PUBLIC_TURN_EVENTS_PROBE),
    VerifierSpec("jev6-v-resolve-available-skill", ("pico/memory_engine/skill_local/registry.py",), "tests/test_skill_forge_local_pool.py", _RESOLVE_AVAILABLE_PROBE),
)


def verifier_set_digest() -> str:
    return canonical_digest(tuple((spec.verifier_id, spec.digest) for spec in SPECS))


def spec_by_id(verifier_id: str) -> VerifierSpec:
    return next(spec for spec in SPECS if spec.verifier_id == verifier_id)


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607 -- repository-native executable
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(sorted(line[3:].replace("\\", "/") for line in completed.stdout.splitlines()))


def verify_task(task_id: str, workspace: Path, *, python_executable: str | None = None) -> VerifierResult:
    task = next(task for task in TASKS if task.task_id == task_id)
    spec = spec_by_id(task.verifier_id)
    findings: list[str] = []
    changed = _changed_paths(workspace)
    allowed = (*spec.production_paths, spec.test_path)
    if any(not any(fnmatch.fnmatch(path, pattern) for pattern in allowed) for path in changed):
        findings.append("prohibited_change_detected")
    if any(not any(fnmatch.fnmatch(path, pattern) for path in changed) for pattern in spec.production_paths):
        findings.append("required_production_change_missing")
    if spec.test_path not in changed:
        findings.append("required_test_change_missing")
    executable = python_executable or sys.executable
    probe = subprocess.run(
        [executable, "-c", spec.semantic_probe],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if probe.returncode != 0:
        findings.append("semantic_probe_failed")
    tests = subprocess.run(
        [executable, "-m", "pytest", spec.test_path, "-q"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if tests.returncode != 0:
        findings.append("targeted_test_failed")
    return VerifierResult(not findings, tuple(findings), spec.verifier_id, spec.digest)


__all__ = ["SPECS", "VerifierResult", "VerifierSpec", "spec_by_id", "verifier_set_digest", "verify_task"]
