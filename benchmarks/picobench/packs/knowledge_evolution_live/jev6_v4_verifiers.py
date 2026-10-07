"""Independent semantic verifiers for the final JEV.6 held-out v4 suite."""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest

from .jev6_v3_verifiers import SPECS as V3_SPECS
from .jev6_v4_suite import NEW_TASK_IDS, TASKS
from .jev6_verifiers import VerifierResult, VerifierSpec

_GIT_EXECUTABLE = shutil.which("git")
if _GIT_EXECUTABLE is None:
    raise RuntimeError("git executable is required for JEV.6V4 verification")

_REPLACE_PROVIDER_MODELS_PROBE = r'''
import json
import tempfile
from pathlib import Path
from pico.config.update_providers import replace_provider_models
with tempfile.TemporaryDirectory() as root:
    path=Path(root)/"config.json"
    path.write_text(json.dumps({"providers":{"openai":{"api_key":"secret","models":["old"]},"anthropic":{"api_key":"other"}}}),encoding="utf-8")
    assert replace_provider_models("openai",[" model-a ","model-b","model-a"],config_path=path)==["model-a","model-b"]
    saved=json.loads(path.read_text(encoding="utf-8"))
    assert saved["providers"]["openai"]["apiKey"]=="secret"
    assert saved["providers"]["openai"]["models"]==["model-a","model-b"]
    assert saved["providers"]["anthropic"]=={"api_key":"other"}
    before=path.read_bytes()
    for invalid in ([""],["   "],["ok",3]):
        try: replace_provider_models("openai",invalid,config_path=path)
        except ValueError: pass
        else: raise AssertionError("invalid model accepted")
        assert path.read_bytes()==before
    try: replace_provider_models("missing",["model"],config_path=path)
    except KeyError: pass
    else: raise AssertionError("unknown provider accepted")
'''

_CAPABILITY_TOKEN_PAYLOAD_PROBE = r'''
import hashlib
import hmac
from pico.auth.capability_token import CapabilityToken, _b64, issue_token, verify_token
material="fixture-signing-material"
valid=CapabilityToken(agent_id="agent",capabilities=["read"],issued_at=1)
assert verify_token(issue_token(valid,material),material)==valid
payload=_b64(b"[]")
signature=_b64(hmac.new(material.encode("ascii"),payload.encode("ascii"),hashlib.sha256).digest())
assert verify_token(f"{payload}.{signature}",material) is None
assert verify_token("\N{SNOWMAN}.bad",material) is None
'''  # noqa: S105 -- deterministic fixture source, not a credential

_PORTABLE_MEDIA_NAME_PROBE = r'''
from pico.channels.media import safe_name
assert safe_name("../../secret.txt")=="secret.txt"
assert safe_name(r"..\..\secret.txt")=="secret.txt"
assert safe_name("folder\\report.pdf")=="report.pdf"
assert safe_name("report.pdf")=="report.pdf"
assert safe_name("")==safe_name(None)==safe_name(".")==safe_name("..") == "file"
'''

_V3_BY_TASK = {
    task.task_id: next(spec for spec in V3_SPECS if spec.verifier_id == task.verifier_id)
    for task in __import__(
        "benchmarks.picobench.packs.knowledge_evolution_live.jev6_v3_suite",
        fromlist=["TASKS"],
    ).TASKS
}
_NEW_SPECS = (
    VerifierSpec(
        "jev6v4-v-replace-provider-models",
        ("pico/config/update_providers.py",),
        "tests/test_config_update_providers.py",
        _REPLACE_PROVIDER_MODELS_PROBE,
    ),
    VerifierSpec(
        "jev6v4-v-capability-token-payload",
        ("pico/auth/capability_token.py",),
        "tests/test_auth_allowlist.py",
        _CAPABILITY_TOKEN_PAYLOAD_PROBE,
    ),
    VerifierSpec(
        "jev6v4-v-portable-media-name",
        ("pico/channels/media.py",),
        "tests/test_channels_media.py",
        _PORTABLE_MEDIA_NAME_PROBE,
    ),
)
_NEW_BY_ID = dict(zip(NEW_TASK_IDS, _NEW_SPECS, strict=True))
SPECS = tuple(
    _NEW_BY_ID.get(task.task_id, _V3_BY_TASK.get(task.task_id))
    for task in TASKS
)
if any(spec is None for spec in SPECS):
    raise RuntimeError("JEV.6 V4 verifier coverage is incomplete")


def verifier_set_digest() -> str:
    ordered = tuple(
        (
            task.verifier_id,
            next(spec.digest for spec in SPECS if spec.verifier_id == task.verifier_id),
        )
        for task in TASKS
    )
    return canonical_digest(ordered)


def spec_by_id(verifier_id: str) -> VerifierSpec:
    return next(spec for spec in SPECS if spec.verifier_id == verifier_id)


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        [_GIT_EXECUTABLE, "status", "--porcelain"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(sorted(line[3:].replace("\\", "/") for line in completed.stdout.splitlines()))


def verify_task(
    task_id: str,
    workspace: Path,
    *,
    python_executable: str | None = None,
) -> VerifierResult:
    task = next(task for task in TASKS if task.task_id == task_id)
    spec = spec_by_id(task.verifier_id)
    findings: list[str] = []
    changed = _changed_paths(workspace)
    allowed = (*spec.production_paths, spec.test_path)
    if any(not any(fnmatch.fnmatch(path, pattern) for pattern in allowed) for path in changed):
        findings.append("prohibited_change_detected")
    if any(
        not any(fnmatch.fnmatch(path, pattern) for path in changed)
        for pattern in spec.production_paths
    ):
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


__all__ = ["SPECS", "spec_by_id", "verifier_set_digest", "verify_task"]
