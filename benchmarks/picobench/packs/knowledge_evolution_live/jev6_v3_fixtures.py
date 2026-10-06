"""Sealed mechanical references for the JEV.6 held-out v3 suite."""

from __future__ import annotations

from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest

from . import jev6_v2_fixtures as v2_fixtures
from .jev6_v3_suite import NEW_TASK_ID, SUITE_VERSION, TASKS


def _provider_resolution(root: Path) -> None:
    path = root / "pico/providers/registry.py"
    source = path.read_text(encoding="utf-8")
    anchor = '\ndef find_by_name(name: str) -> ProviderSpec | None:\n'
    if anchor not in source:
        raise RuntimeError(f"JEV.6 v3 reference anchor missing: {path}")
    function = '''
def resolve_provider_spec(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderSpec | None:
    if provider_name and (exact := find_by_name(provider_name)) is not None:
        return exact
    gateway = find_gateway(
        provider_name=provider_name,
        api_key=api_key,
        api_base=api_base,
    )
    if gateway is not None:
        return gateway
    return find_by_model(model) if model else None

'''
    path.write_text(source.replace(anchor, function + anchor, 1), encoding="utf-8")
    test = root / "tests/test_provider_catalog.py"
    test.write_text(
        test.read_text(encoding="utf-8")
        + f"\n\n# Sealed JEV.6 v3 mechanical reference: {NEW_TASK_ID}.\n",
        encoding="utf-8",
    )


def apply_reference(task_id: str, workspace: Path) -> None:
    if task_id == NEW_TASK_ID:
        _provider_resolution(workspace)
        return
    v2_fixtures.apply_reference(task_id, workspace)


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
