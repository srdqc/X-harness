"""Sealed mechanical references for the final JEV.6 held-out v4 suite."""

from __future__ import annotations

from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest

from . import jev6_v3_fixtures as v3_fixtures
from .jev6_v4_suite import NEW_TASK_IDS, SUITE_VERSION, TASKS


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if old not in source:
        raise RuntimeError(f"JEV.6 v4 reference anchor missing: {path}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _replace_provider_models(root: Path) -> None:
    path = root / "pico/config/update_providers.py"
    anchor = "\ndef add_provider_model(\n"
    implementation = '''
def replace_provider_models(
    name: str,
    models: list[str],
    *,
    config_path: Path | None = None,
) -> list[str]:
    """Atomically replace a Provider's curated model list."""

    normalized: list[str] = []
    for model in models:
        if not isinstance(model, str) or not (value := model.strip()):
            raise ValueError("provider model names must be non-empty strings")
        if value not in normalized:
            normalized.append(value)
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    cls = _provider_schema_cls(name)
    section = dict((data.get("providers") or {}).get(name) or {})
    section["models"] = normalized
    validated = cls.model_validate(section)
    data.setdefault("providers", {})
    data["providers"][name] = validated.model_dump(by_alias=True)
    _write_atomic(path, data)
    return normalized

'''
    _replace_once(path, anchor, "\n" + implementation + "def add_provider_model(\n")
    _replace_once(
        path,
        '__all__ = [\n    "provider_field_specs",',
        '__all__ = [\n    "replace_provider_models",\n    "provider_field_specs",',
    )
    test = root / "tests/test_config_update_providers.py"
    test.write_text(
        test.read_text(encoding="utf-8")
        + '''

def test_replace_provider_models_normalizes_and_preserves_fields(cfg_path: Path) -> None:
    from pico.config.update_providers import get_provider_config, replace_provider_models, set_provider_fields

    set_provider_fields("openai", {"api_key": "secret"}, config_path=cfg_path)
    assert replace_provider_models(
        "openai", [" gpt-a ", "gpt-b", "gpt-a"], config_path=cfg_path
    ) == ["gpt-a", "gpt-b"]
    assert get_provider_config("openai", redact_secrets=False, config_path=cfg_path)["api_key"] == "secret"
''',
        encoding="utf-8",
    )


def _capability_token_payload(root: Path) -> None:
    path = root / "pico/auth/capability_token.py"
    old = '''    payload_b64, sig_b64 = token_str.split(".", 1)
    expected = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), _SIG_ALGO)
    if not hmac.compare_digest(_b64(expected.digest()), sig_b64):
        return None
    try:
        raw = _unb64(payload_b64)
        payload = json.loads(raw.decode("utf-8"))
        token = CapabilityToken.from_payload(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
'''
    new = '''    payload_b64, sig_b64 = token_str.split(".", 1)
    try:
        expected = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), _SIG_ALGO)
        if not hmac.compare_digest(_b64(expected.digest()), sig_b64):
            return None
        raw = _unb64(payload_b64)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        token = CapabilityToken.from_payload(payload)
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
'''
    _replace_once(path, old, new)
    test = root / "tests/test_auth_allowlist.py"
    test.write_text(
        test.read_text(encoding="utf-8")
        + '''

def test_capability_token_signed_non_object_fails_closed() -> None:
    import hashlib
    import hmac
    from pico.auth.capability_token import _b64

    payload = _b64(b"[]")
    signature = _b64(hmac.new(b"secret", payload.encode("ascii"), hashlib.sha256).digest())
    assert verify_token(f"{payload}.{signature}", "secret") is None
    assert verify_token("\N{SNOWMAN}.bad", "secret") is None
''',
        encoding="utf-8",
    )


def _portable_media_name(root: Path) -> None:
    path = root / "pico/channels/media.py"
    old = '    return os.path.basename(name or "") or "file"\n'
    new = '''    normalized = (name or "").replace("\\\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    return basename if basename not in {"", ".", ".."} else "file"
'''
    _replace_once(path, old, new)
    test = root / "tests/test_channels_media.py"
    test.write_text(
        test.read_text(encoding="utf-8")
        + '''

def test_safe_name_is_portable_across_path_styles():
    assert media.safe_name(r"..\\..\\secret.txt") == "secret.txt"
    assert media.safe_name("../../secret.txt") == "secret.txt"
    assert media.safe_name("..") == "file"
    assert media.safe_name(".") == "file"
''',
        encoding="utf-8",
    )


_APPLIERS = {
    NEW_TASK_IDS[0]: _replace_provider_models,
    NEW_TASK_IDS[1]: _capability_token_payload,
    NEW_TASK_IDS[2]: _portable_media_name,
}


def apply_reference(task_id: str, workspace: Path) -> None:
    if task_id in _APPLIERS:
        _APPLIERS[task_id](workspace)
        return
    v3_fixtures.apply_reference(task_id, workspace)


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
