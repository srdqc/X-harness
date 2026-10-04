from __future__ import annotations

import json

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live import jev5a_probe
from pico.providers.base import LLMProvider, LLMResponse


class _ProbeProvider(LLMProvider):
    _CHAT_RETRY_DELAYS = ()

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get_default_model(self) -> str:
        return jev5a_probe.MODEL_ID

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append(
            {"messages": messages, "tools": tools, "model": model, **kwargs}
        )
        return self.responses.pop(0)


def _response(choices: tuple[str, ...]) -> LLMResponse:
    content = json.dumps(
        {
            "decisions": [
                {"candidate": f"candidate_{index}", "choice": choice}
                for index, choice in enumerate(choices)
            ]
        }
    )
    return LLMResponse(
        content=content,
        finish_reason="stop",
        model="deepseek-flash",
        usage={
            "prompt_tokens": 40,
            "completion_tokens": 12,
            "reasoning_tokens": 0,
        },
    )


@pytest.mark.asyncio
async def test_probe_is_exactly_three_tool_free_structured_utility_calls() -> None:
    provider = _ProbeProvider(
        [_response(("KEEP",)), _response(("ABSTAIN",)), _response(("KEEP", "ABSTAIN"))]
    )
    artifact = await jev5a_probe.run_probe(provider)
    jev5a_probe.validate_artifact(artifact)

    assert len(provider.calls) == 3
    assert artifact["planned_logical_calls"] == 3
    assert artifact["actual_logical_calls"] == 3
    assert artifact["contract_success_count"] == 3
    assert artifact["fallback_count"] == 0
    assert [item["raw_choices"] for item in artifact["results"]] == [
        ("KEEP",),
        ("ABSTAIN",),
        ("KEEP", "ABSTAIN"),
    ]
    for call in provider.calls:
        assert call["tools"] is None
        assert call["reasoning_effort"] == "none"
        assert call["response_format"] == {"type": "json_object"}
        assert call["max_tokens"] == 1024
        serialized = json.dumps(call, ensure_ascii=False)
        assert "expected_direction" not in serialized
        assert "P1/C" not in serialized and "P2/C" not in serialized and "P3/C" not in serialized


@pytest.mark.asyncio
async def test_probe_failure_is_recorded_without_replacement_call() -> None:
    provider = _ProbeProvider(
        [
            LLMResponse(
                content="",
                finish_reason="length",
                model="deepseek-flash",
                usage={"completion_tokens": 1024, "reasoning_tokens": 1024},
            ),
            _response(("ABSTAIN",)),
            _response(("KEEP", "ABSTAIN")),
        ]
    )
    artifact = await jev5a_probe.run_probe(provider)

    assert len(provider.calls) == 3
    assert artifact["actual_logical_calls"] == 3
    assert artifact["contract_success_count"] == 2
    assert artifact["fallback_count"] == 1
    failed = artifact["results"][0]
    assert failed["generation_outcome"] == "LENGTH"
    assert failed["payload_outcome"] == "EMPTY_CONTENT"
    assert failed["reasoning_tokens"] == 1024
    assert failed["visible_output_tokens"] == 0


def test_probe_artifact_validator_rejects_call_budget_overrun() -> None:
    with pytest.raises(ValueError, match="logical-call bound"):
        jev5a_probe.validate_artifact(
            {
                "schema": jev5a_probe.PROBE_SCHEMA,
                "actual_logical_calls": 4,
                "results": tuple(
                    {"case_id": case.case_id} for case in jev5a_probe.CASES
                ),
            }
        )
