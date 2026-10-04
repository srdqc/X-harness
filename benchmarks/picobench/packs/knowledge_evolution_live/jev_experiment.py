"""Frozen design contract for a future, separately authorized Jev utility study."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.picobench.canonical import canonical_digest
from pico.decision_plane.provider_utility import (
    PROVIDER_UTILITY_POLICY_VERSION,
    PROVIDER_UTILITY_PROMPT_DIGEST,
)

JEV_UTILITY_EXPERIMENT_SCHEMA = "pico.picobench.jev-utility-exploratory.v1"
JEV_UTILITY_EXPERIMENT_ARMS = (
    "no_reuse",
    "task_relevance_v1",
    "task_relevance_v1_jev_utility",
)
JEV_UTILITY_MAIN_COMPARISON = (
    "task_relevance_v1",
    "task_relevance_v1_jev_utility",
)
JEV_UTILITY_PROVIDER_ARM = "task_relevance_v1_provider_utility"
JEV_UTILITY_EXPERIMENT_QUESTIONS = (
    "Does Jev utility gating preserve verified success relative to TASK_RELEVANCE_V1?",
    "Does it reduce unnecessary candidate injection?",
    "Does it reduce Provider calls, Tools, tokens, reads, or latency when relevance selected candidates?",
    "Does it skip zero-selection tasks without invoking Jev?",
    "Does it avoid harming tasks where deterministic selective reuse was already beneficial?",
    "How often does Jev abstain?",
    "How often does fallback occur?",
    "Does Jev decision latency outweigh downstream savings?",
)


@dataclass(frozen=True)
class JevUtilityExperimentDesign:
    arms: tuple[str, ...] = JEV_UTILITY_EXPERIMENT_ARMS
    main_comparison: tuple[str, str] = JEV_UTILITY_MAIN_COMPARISON
    questions: tuple[str, ...] = JEV_UTILITY_EXPERIMENT_QUESTIONS
    schema: str = JEV_UTILITY_EXPERIMENT_SCHEMA
    version: int = 1

    def treatment_identity(
        self,
        *,
        provider_id: str,
        model_id: str,
        timeout_seconds: float,
        max_candidates: int,
    ) -> dict[str, object]:
        """Secret-free identity for future arm C; no campaign or Provider call is created."""

        return {
            "arm": JEV_UTILITY_PROVIDER_ARM,
            "utility_backend": "provider",
            "utility_provider": provider_id,
            "utility_model": model_id,
            "typed_choice_policy_version": PROVIDER_UTILITY_POLICY_VERSION,
            "timeout_seconds": timeout_seconds,
            "max_candidates": max_candidates,
            "prompt_template_digest": PROVIDER_UTILITY_PROMPT_DIGEST,
            "response_schema": "pico.jev-utility-response.v1",
            "response_schema_version": 1,
        }

    def treatment_config_digest(self, **kwargs: object) -> str:
        return canonical_digest(self.treatment_identity(**kwargs))

    @property
    def design_digest(self) -> str:
        return canonical_digest(self)


__all__ = [
    "JEV_UTILITY_EXPERIMENT_ARMS",
    "JEV_UTILITY_EXPERIMENT_QUESTIONS",
    "JEV_UTILITY_EXPERIMENT_SCHEMA",
    "JEV_UTILITY_MAIN_COMPARISON",
    "JEV_UTILITY_PROVIDER_ARM",
    "JevUtilityExperimentDesign",
]
