from __future__ import annotations

from typing import Protocol

from benchmarks.picobench.fixtures.mcp import (
    MCP_CATALOG_SIZE,
    catalog_digest,
)
from benchmarks.picobench.fixtures.mcp.live_catalog import (
    LIVE_MCP_CATALOG_SIZE,
    live_catalog_digest,
)
from benchmarks.picobench.protocol import TrialContext, TrialExecution
from benchmarks.picobench.records import (
    TrialStatus,
    VerificationState,
    VerifierResult,
)
from benchmarks.picobench.schema import PackDefinition, PairSpec, VariantSpec

from .live_verifier import live_verifier_code_digest
from .metrics import (
    TOOL_SCHEMA_ESTIMATOR_DIGEST,
    TOOL_SCHEMA_ESTIMATOR_ID,
)
from .models import ToolMCPTask, ToolMCPTrack
from .runner import TOOL_MCP_MAX_TOOL_ITERATIONS
from .tasks import load_tool_mcp_tasks, tool_mcp_task_set_digest
from .verifier import mcp_verifier_code_digest

_REQUIRED_METRICS = frozenset(
    {
        "mcp_transport",
        "mcp_connected",
        "mcp_catalog_count",
        "mcp_catalog_digest",
        "mcp_catalog_source_digest",
        "mcp_verifier_digest",
        "mcp_receipt_count",
        "initial_visible_tool_count",
        "initial_visible_catalog_tool_count",
        "visible_tool_schema_tokens_per_call",
        "trial_total_estimated_visible_tool_schema_tokens",
        "tool_array_schema_tokens_per_call",
        "cumulative_tool_array_schema_tokens",
        "in_band_disclosure_tokens_per_call",
        "cumulative_in_band_disclosure_tokens",
        "total_disclosure_proxy",
        "unique_in_band_disclosure_tokens_per_call",
        "cumulative_unique_in_band_disclosure_tokens",
        "provider_visible_in_band_disclosure_tokens_per_call",
        "cumulative_provider_visible_in_band_disclosure_tokens",
        "cumulative_provider_visible_disclosure_tokens",
        "fallback_count",
        "fallback_rate",
        "fallback_reason_counts",
        "provider_attempts_before_fallback",
        "provider_attempts_after_fallback",
        "success_after_fallback",
        "average_visible_tool_count",
        "peak_visible_tool_count",
        "schema_estimator_id",
        "schema_estimator_digest",
        "meta_tool_invocations",
        "meta_tool_failures",
        "first_target_accuracy",
        "invalid_target_call_rate",
        "exact_target_repeat_rate",
        "normalized_target_call_count",
        "target_call_records",
        "tool_retrieval_query_count",
        "tool_retrieval_evidence",
        "target_recall_at_1",
        "target_recall_at_3",
        "target_recall_at_5",
        "first_target_retrieval_rank",
        "target_retrieval_mrr",
        "zero_hit_count",
        "zero_hit_rate",
        "wrong_target_ranking_evidence",
        "provider_model",
        "actual_model_names",
        "generation_settings",
        "provider_call_max_attempts",
        "model_call_records",
        "main_agent_model_calls",
        "main_agent_input_tokens",
        "main_agent_output_tokens",
        "main_agent_total_tokens",
        "main_agent_cache_read_tokens",
        "main_agent_cache_write_tokens",
        "main_agent_reasoning_tokens",
        "main_agent_cost_usd",
        "usage_complete",
        "cost_complete",
        "end_to_end_latency_ms",
        "selected_role",
        "role_prompt_tokens_per_call",
        "cumulative_role_prompt_tokens",
        "off_role_tool_selection_count",
        "off_role_tool_selection_rate",
        "irrelevant_ranked_before_target_count",
        "evidence_before_first_mutation",
        "verification_after_mutation",
        "repository_inspection_before_edit",
        "verification_after_edit",
        "unnecessary_mutation_count",
    }
)


class ToolMCPTrialRunner(Protocol):
    async def __call__(
        self,
        *,
        context: TrialContext,
        task: ToolMCPTask,
    ) -> TrialExecution: ...


class ToolMCPPack:
    def __init__(
        self,
        track: ToolMCPTrack = ToolMCPTrack.FORMAL,
        *,
        runner: ToolMCPTrialRunner | None = None,
    ) -> None:
        self.track = ToolMCPTrack(track)
        self._runner = runner
        self._tasks = {task.task_id: task for task in load_tool_mcp_tasks(self.track)}

    def definition(self) -> PackDefinition:
        if self.track is ToolMCPTrack.LIVE_SOLVABLE_V1:
            return self._live_solvable_definition()
        if self.track is ToolMCPTrack.ROLE_EXPERIMENT:
            return self._role_experiment_definition()
        pack_id = "tool-mcp" if self.track is ToolMCPTrack.FORMAL else "tool-mcp-calibration"
        return PackDefinition(
            pack_id=pack_id,
            tasks=tuple(task.to_task_spec() for task in self._tasks.values()),
            variants=(
                VariantSpec(
                    variant_id="tool-mcp-all-tools",
                    settings={"tool_disclosure": "all_tools"},
                ),
                VariantSpec(
                    variant_id="tool-mcp-progressive",
                    settings={
                        "tool_disclosure": "progressive_disclosure",
                    },
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="tool_disclosure",
                    control_variant_id="tool-mcp-all-tools",
                    treatment_variant_id="tool-mcp-progressive",
                ),
            ),
            identity={
                "claim_reducer": "tool_mcp_v1",
                "task_set_digest": tool_mcp_task_set_digest(self.track),
                "mcp_catalog_digest": catalog_digest(),
                "mcp_verifier_digest": mcp_verifier_code_digest(),
                "mcp_catalog_size": MCP_CATALOG_SIZE,
                "mcp_transport": "stdio",
                "schema_estimator_id": TOOL_SCHEMA_ESTIMATOR_ID,
                "schema_estimator_digest": TOOL_SCHEMA_ESTIMATOR_DIGEST,
                "max_tool_iterations": TOOL_MCP_MAX_TOOL_ITERATIONS,
                "result_scope": (
                    "exploratory_eight_task_pack" if self.track is ToolMCPTrack.FORMAL else "calibration_only"
                ),
            },
        )

    def _role_experiment_definition(self) -> PackDefinition:
        variants = (
            VariantSpec(
                variant_id="role-full",
                settings={
                    "tool_disclosure": "all_tools",
                    "role_prompt": False,
                    "role_prior": False,
                },
            ),
            VariantSpec(
                variant_id="role-prog",
                settings={
                    "tool_disclosure": "progressive_disclosure",
                    "role_prompt": False,
                    "role_prior": False,
                },
            ),
            VariantSpec(
                variant_id="role-prompt",
                settings={
                    "tool_disclosure": "progressive_disclosure",
                    "role_prompt": True,
                    "role_prior": False,
                },
            ),
            VariantSpec(
                variant_id="role-aware",
                settings={
                    "tool_disclosure": "progressive_disclosure",
                    "role_prompt": True,
                    "role_prior": True,
                },
            ),
        )
        return PackDefinition(
            pack_id="tool-mcp-role-experiment",
            tasks=tuple(task.to_task_spec() for task in self._tasks.values()),
            variants=variants,
            pairs=(
                PairSpec(
                    treatment_axis="tool_disclosure",
                    control_variant_id=variants[0].variant_id,
                    treatment_variant_id=variants[1].variant_id,
                ),
                PairSpec(
                    treatment_axis="role_prompt",
                    control_variant_id=variants[1].variant_id,
                    treatment_variant_id=variants[2].variant_id,
                ),
                PairSpec(
                    treatment_axis="role_prior",
                    control_variant_id=variants[2].variant_id,
                    treatment_variant_id=variants[3].variant_id,
                ),
            ),
            identity={
                "claim_reducer": "experimental_role_v1",
                "task_set_digest": tool_mcp_task_set_digest(self.track),
                "mcp_catalog_digest": catalog_digest(),
                "mcp_verifier_digest": mcp_verifier_code_digest(),
                "mcp_catalog_size": MCP_CATALOG_SIZE,
                "mcp_transport": "stdio",
                "schema_estimator_id": TOOL_SCHEMA_ESTIMATOR_ID,
                "schema_estimator_digest": TOOL_SCHEMA_ESTIMATOR_DIGEST,
                "max_tool_iterations": TOOL_MCP_MAX_TOOL_ITERATIONS,
                "result_scope": "experimental_role_comparison",
            },
        )

    def _live_solvable_definition(self) -> PackDefinition:
        variants = (
            VariantSpec(
                variant_id="live-full",
                settings={
                    "tool_disclosure": "all_tools",
                    "role_prompt": False,
                    "role_prior": False,
                },
            ),
            VariantSpec(
                variant_id="live-progressive",
                settings={
                    "tool_disclosure": "progressive_disclosure",
                    "role_prompt": False,
                    "role_prior": False,
                },
            ),
            VariantSpec(
                variant_id="live-role-aware",
                settings={
                    "tool_disclosure": "progressive_disclosure",
                    "role_prompt": True,
                    "role_prior": True,
                },
            ),
        )
        return PackDefinition(
            pack_id="tool-mcp-live-solvable-v1",
            tasks=tuple(task.to_task_spec() for task in self._tasks.values()),
            variants=variants,
            pairs=(
                PairSpec(
                    treatment_axis="tool_disclosure",
                    control_variant_id=variants[0].variant_id,
                    treatment_variant_id=variants[1].variant_id,
                ),
                PairSpec(
                    treatment_axis="role_guidance",
                    control_variant_id=variants[1].variant_id,
                    treatment_variant_id=variants[2].variant_id,
                ),
            ),
            identity={
                "claim_reducer": "live_solvable_v1",
                "task_set_digest": tool_mcp_task_set_digest(self.track),
                "mcp_catalog_digest": live_catalog_digest(),
                "mcp_verifier_digest": live_verifier_code_digest(),
                "mcp_catalog_size": LIVE_MCP_CATALOG_SIZE,
                "mcp_transport": "stdio",
                "schema_estimator_id": TOOL_SCHEMA_ESTIMATOR_ID,
                "schema_estimator_digest": TOOL_SCHEMA_ESTIMATOR_DIGEST,
                "max_tool_iterations": TOOL_MCP_MAX_TOOL_ITERATIONS,
                "result_scope": "live_model_acceptance_v1",
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        if self._runner is None:
            return _infrastructure_failure(
                context,
                "tool_mcp_trial_runner_not_configured",
            )
        task = self._tasks.get(context.task.task_id)
        if task is None:
            return _infrastructure_failure(context, "unknown_tool_mcp_task")
        disclosure = context.variant.settings.get("tool_disclosure")
        if disclosure not in {"all_tools", "progressive_disclosure"}:
            return _infrastructure_failure(
                context,
                "tool_disclosure_not_supported",
            )
        execution = await self._runner(context=context, task=task)
        missing = sorted(_REQUIRED_METRICS - execution.metrics.keys())
        if missing:
            return _infrastructure_failure(
                context,
                "missing_tool_mcp_metrics:" + ",".join(missing),
            )
        if dict(execution.observed_variant_settings) != dict(context.variant.settings):
            return _infrastructure_failure(
                context,
                "observed_variant_settings_drift",
            )
        return execution


def _infrastructure_failure(
    context: TrialContext,
    finding: str,
) -> TrialExecution:
    return TrialExecution(
        status=TrialStatus.INFRASTRUCTURE_FAILURE,
        runtime_state=None,
        delivery_state=None,
        verification=VerifierResult(
            state=VerificationState.NOT_RUN,
            findings=(finding,),
        ),
        observed_variant_settings=dict(context.variant.settings),
        findings=(finding,),
    )


__all__ = ["ToolMCPPack", "ToolMCPTrialRunner"]
