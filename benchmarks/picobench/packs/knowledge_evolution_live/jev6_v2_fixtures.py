"""Sealed mechanical references for the JEV.6 held-out v2 suite."""

from __future__ import annotations

from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest

from . import jev6_fixtures as v1_fixtures
from .jev6_v2_suite import NEW_TASK_ID, SUITE_VERSION, TASKS


def _plugin_contributions(root: Path) -> None:
    path = root / "pico/plugin/registry.py"
    source = path.read_text(encoding="utf-8")
    anchor = "    def activated_ids(self) -> list[str]:\n"
    if anchor not in source:
        raise RuntimeError(f"JEV.6 v2 reference anchor missing: {path}")
    method = '''    def contributions_for(self, plugin_id: str) -> tuple[tuple[str, str, str], ...]:
        if plugin_id not in self._manifests:
            return ()
        values = (
            *(("memory_backend", name, entry.factory_ref) for name, entry in self._memory_backends.items() if entry.plugin_id == plugin_id),
            *(("tool", name, entry.factory_ref) for name, entry in self._tools.items() if entry.plugin_id == plugin_id),
        )
        return tuple(sorted(values, key=lambda item: (item[0], item[1])))

'''
    path.write_text(source.replace(anchor, method + anchor, 1), encoding="utf-8")
    test = root / "tests/test_plugin_registry.py"
    test.write_text(
        test.read_text(encoding="utf-8")
        + f"\n\n# Sealed JEV.6 v2 mechanical reference: {NEW_TASK_ID}.\n",
        encoding="utf-8",
    )


def apply_reference(task_id: str, workspace: Path) -> None:
    if task_id == NEW_TASK_ID:
        _plugin_contributions(workspace)
        return
    v1_fixtures.apply_reference(task_id, workspace)


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
