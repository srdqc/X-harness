from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live import jev3b_pack
from pico.decision_plane.provider_utility import ProviderUtilityBackend, ProviderUtilityConfig
from pico.decision_plane.utility import JevUtilityDecisionAdapter
from pico.providers.base import LLMProvider, LLMResponse


class _PackProvider(LLMProvider):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content
        self.calls: list[dict] = []

    def get_default_model(self) -> str:
        return jev3b_pack.MODEL_ID

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "model": model, **kwargs})
        return LLMResponse(
            content=self.content,
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            model=model,
        )


def test_pack_is_exactly_eight_frozen_cases_with_deterministic_digest() -> None:
    assert len(jev3b_pack.PACK_CASES) == 8
    assert sum(len(item.candidates) for item in jev3b_pack.PACK_CASES) == 10
    assert jev3b_pack.MAX_LOGICAL_CALLS == 8
    assert jev3b_pack.pack_manifest() == jev3b_pack.pack_manifest()
    assert len(jev3b_pack.pack_manifest()["semantic_digest"]) == 64
    assert jev3b_pack.pack_manifest()["freeze_digest"] == jev3b_pack.pack_manifest()[
        "semantic_digest"
    ]


def test_expected_labels_and_case_categories_never_enter_provider_message() -> None:
    provider = _PackProvider('{"decisions":[]}')
    backend = ProviderUtilityBackend(
        provider,
        ProviderUtilityConfig(
            provider_id=jev3b_pack.PROVIDER_ID,
            model_id=jev3b_pack.MODEL_ID,
        ),
    )
    for case in jev3b_pack.PACK_CASES:
        request = jev3b_pack.build_request(case)
        aliases = tuple(f"candidate_{index}" for index in range(len(request.candidates)))
        serialized = json.dumps(backend._messages(request, aliases), ensure_ascii=False)
        assert "handcrafted_expectation" not in serialized
        assert "expectation_reason" not in serialized
        assert case.case_id not in serialized
        assert all(candidate.expectation_reason not in serialized for candidate in case.candidates)
        assert all(candidate.preconditions in serialized for candidate in case.candidates)


def test_pack_does_not_reuse_official_p3r_identifiers_or_labels() -> None:
    serialized = json.dumps(jev3b_pack.semantic_payload(), ensure_ascii=False).casefold()
    for forbidden in (
        "p3r4",
        "nav-02",
        "nav-03",
        "p3r-543e189ff8cfbbad",
        "p3r-a276191655c89e8f",
        "ground_truth",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_two_candidate_case_still_uses_one_tool_free_logical_call() -> None:
    case = jev3b_pack.PACK_CASES[3]
    assert len(case.candidates) == 2
    provider = _PackProvider(
        json.dumps(
            {
                "decisions": [
                    {"candidate": "candidate_0", "choice": "KEEP", "confidence": 0.8},
                    {"candidate": "candidate_1", "choice": "ABSTAIN", "confidence": 0.8},
                ]
            }
        )
    )
    backend = ProviderUtilityBackend(
        provider,
        ProviderUtilityConfig(
            provider_id=jev3b_pack.PROVIDER_ID,
            model_id=jev3b_pack.MODEL_ID,
            max_candidates=2,
        ),
    )
    result = await JevUtilityDecisionAdapter(backend, timeout_seconds=15).decide(
        jev3b_pack.build_request(case)
    )
    assert result.logical_calls == 1
    assert len(provider.calls) == 1
    assert provider.calls[0]["tools"] is None


def test_reducer_scores_raw_choices_and_fallback_separately() -> None:
    results = []
    for case in jev3b_pack.PACK_CASES:
        decisions = [
            {
                "candidate_id": candidate.candidate_id,
                "raw_choice": candidate.handcrafted_expectation,
                "effective_decision": (
                    "KEEP"
                    if candidate.handcrafted_expectation == "UNCERTAIN"
                    else candidate.handcrafted_expectation
                ),
                "confidence": 0.75,
            }
            for candidate in case.candidates
        ]
        results.append(
            {
                "case_id": case.case_id,
                "decision_source": "MODEL_DECISION",
                "schema_valid": True,
                "decisions": decisions,
                "fallback_used": False,
                "provider_logical_calls": 1,
                "provider_attempts": 1,
                "input_tokens": 10,
                "output_tokens": 2,
                "provider_latency_ms": 100.0,
                "receipt_produced": True,
                "tool_call_count": 0,
                "normal_agent_turn_count": 0,
                "workspace_mutation": False,
                "secret_leak": False,
            }
        )
    summary = jev3b_pack.reduce_results(results)
    assert summary["contract_status"] == "PASS"
    assert summary["behavioral_signal"] == "PROMISING"
    assert summary["handcrafted_case_exact_match_count"] == 8
    assert summary["handcrafted_decision_exact_match_count"] == 10
    assert summary["raw_choice_counts"] == {"KEEP": 4, "ABSTAIN": 4, "UNCERTAIN": 2}

    fallback = dict(results[-1])
    fallback.update(decision_source="FALLBACK", decisions=[], fallback_used=True)
    fallback_summary = jev3b_pack.reduce_results([*results[:-1], fallback])
    assert fallback_summary["fallback_case_count"] == 1
    assert fallback_summary["handcrafted_case_exact_match_count"] == 7
    assert fallback_summary["per_choice_agreement"]["KEEP"]["FALLBACK"] == 1


def test_pack_runner_has_no_agent_tool_or_skill_activation_surface() -> None:
    source = Path(jev3b_pack.__file__).read_text(encoding="utf-8")
    assert "AgentLoop" not in source
    assert "ToolRegistry" not in source
    assert "activate_skill" not in source
    assert jev3b_pack.reduce_results([])["skill_activation_count"] == 0
    assert jev3b_pack.reduce_results([])["normal_agent_turn_count"] == 0
    assert jev3b_pack.reduce_results([])["tool_call_count"] == 0


def test_prepare_persists_only_frozen_local_manifest(tmp_path: Path) -> None:
    root = jev3b_pack.prepare_artifact(tmp_path)
    assert root.parent == tmp_path / ".p3r"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == jev3b_pack.pack_manifest()
    serialized = json.dumps(manifest).casefold()
    assert "api_key" not in serialized
    assert "authorization" not in serialized
    assert not (root / "run.json").exists()
