"""Pico feature configuration - extends the base Config with feature blocks.

Usage:
    from pico.config import PicoConfig, load_pico_config

    cfg = load_pico_config()
    if cfg.context.engine == "curator":
        ...

Design:
    - ``PicoConfig`` composes the base ``Config`` rather than subclassing
      it. This keeps the base schema untouched and lets us add / remove
      feature blocks without breaking the base loader.
    - Each feature block has its own Pydantic model. Behavior-changing paths
      default off or observe-only so a fresh install does not rewrite requests.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

from pico.config.loader import (
    EXTENSION_KEYS,
    _migrate_config,
    get_config_path,
)
from pico.config.loader import load_config as load_base_config
from pico.config.schema import Config as BaseConfig


class _Base(BaseModel):
    """Accepts both camelCase and snake_case keys.

    ``extra='forbid'`` catches typos at startup. Retired fields with
    known legacy presence are stripped explicitly in
    ``loader._migrate_config`` before Pydantic validates; unlisted
    unknown keys still raise.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


# ---------------------------------------------------------------------------
# 功能 1：上下文管理（Curator）
# ---------------------------------------------------------------------------


class ContextConfig(_Base):
    """Context engine selection and tuning."""

    engine: str = "unified"
    """Deprecated — there is now a single :class:`ContextAssembler`.

    The historical ``"legacy"`` / ``"curator"`` / ``"default"`` split was
    collapsed: every turn runs the Curator history, Memory, and Local Skill
    lanes in one engine. The field is retained (as a
    free string) so existing YAML setting ``engine: legacy`` etc. still
    loads — the value is ignored by ``build_context_engine``.
    """

    # Curator 历史路径参数。
    fast_path_threshold: float = 0.60
    """Curator Fast Path cutoff. Below this % of budget → zero-LLM pass-through."""

    curator_model: str = "gemini-2.5-flash"
    """Model used by the Curator agent loop (Slow Path). Kept small & fast."""

    curator_timeout_seconds: float = 30.0
    """Max wall time for one Curator slow-path invocation before fallback."""

    relevance_decay: float = 0.95
    """Per-turn decay factor for non-recent message relevance."""

    relevance_reference_boost: float = 0.15
    """Boost applied when assistant response references older message content."""

    protect_first_n: int = 3
    """Number of head exchanges always preserved in context."""

    archive_dir: str = "memory/.curator/archive"
    """Relative path under workspace for lossless message archives."""


# 功能 2：调用效率
# ---------------------------------------------------------------------------


class BudgetPolicyConfig(_Base):
    """Per-session / per-day spend limits."""

    warn_at_usd: float = 0.50
    hard_limit_usd: float = 2.00
    warn_at_input_tokens: int = 500_000
    track_per_session: bool = True
    track_global_daily: bool = True


class SmartRoutingConfig(_Base):
    """SmartRouter configuration."""

    enabled: bool = False
    tiers: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "light": ["gemini-2.5-flash", "claude-haiku-4-5"],
            "medium": ["claude-sonnet-4-6", "gpt-4.1-mini"],
            "heavy": ["claude-opus-4-6", "gpt-4.1"],
        }
    )
    default_tier: Literal["light", "medium", "heavy"] = "heavy"
    """Fallback tier when routing is uncertain — conservative default."""


class ToolResultLifecycleConfig(_Base):
    """Tool result lifecycle management (the three-phase pruner)."""

    enabled: bool = False
    full_retention_turns: int = 3
    summary_retention_turns: int = 10
    placeholder_text: str = "[Tool result archived — retrievable via Curator]"
    summary_model: str = "gemini-2.5-flash"


class CallEfficiencyConfig(_Base):
    """Provider-call cache, normalized usage, and estimated-cost policy."""

    mode: Literal["off", "observe", "optimize"] = "observe"
    """Observe by default; request rewriting requires explicit optimize mode."""

    enabled: bool = True
    """Legacy TokenWise switch. False maps to effective mode off."""

    usage_tracking: bool = True
    """Record token usage per call — cheap and informative; on by default."""

    cache_optimization: bool = True
    """Legacy TokenWise field. Canonical request behavior is selected by mode."""

    max_cache_breakpoints: int = 4
    """Anthropic API limit; kept configurable for forward-compat."""

    skill_lazy_loading: bool = False
    """Only inject skill summaries relevant to the current message."""

    tool_result_lifecycle: ToolResultLifecycleConfig = Field(default_factory=ToolResultLifecycleConfig)
    smart_routing: SmartRoutingConfig = Field(default_factory=SmartRoutingConfig)
    budget: BudgetPolicyConfig = Field(default_factory=BudgetPolicyConfig)

    @property
    def effective_mode(self) -> Literal["off", "observe", "optimize"]:
        return self.mode if self.enabled else "off"


TokenWiseConfig = CallEfficiencyConfig


# ---------------------------------------------------------------------------
# 功能 3：SkillForge
# ---------------------------------------------------------------------------
#
# SkillForge 负责 Local Skill 的检索与执行。
#
# 配置有意保持扁平。组件级参数
# （嵌入模型、BM25 参数等）位于
# 暂时留在 ``skill_forge/`` 内的脚手架 dataclass 中并使用默认值；
# 需要向用户开放时，再由负责人把相应字段提升到这里。


class LocalDirConfig(_Base):
    """One local skill directory entry (R1)."""

    path: str
    """Absolute or ``~``-relative path. Expanded at startup."""

    enabled: bool = True
    """False → directory completely skipped."""

    name: str | None = None
    """Display name for logs. None → derived from path basename."""

    always_enabled: bool = True
    """False → skills from this dir with ``always: true`` are excluded
    from always injection (but still retrievable via select)."""


class SkillForgeConfig(_Base):
    """SkillForge configuration.

    The active Runtime resolves operator-managed Local Skills without Provider
    calls. Repository Memory is independent of this subsystem. Legacy remote
    retrieval and LLM-selection fields remain loadable for old evaluation
    configurations but do not participate in the active Runtime path.
    """

    # --- 总开关与位置 ---
    enabled: bool = True
    """Compatibility switch used while discovering configured Local Skill directories."""

    router: "SkillForgeRouterConfig" = Field(
        default_factory=lambda: SkillForgeRouterConfig(),
    )
    """Local BM25 routing policy, under config key ``skillForge.router``. The
    router is a component of the SkillForge subsystem, so it nests here
    rather than living as a sibling top-level block. Forward-ref +
    ``model_rebuild`` (below): ``SkillForgeRouterConfig`` is defined later
    in this module."""

    local_dirs: list[LocalDirConfig] = Field(default_factory=list)
    """Local skill directories to mount (R1). List order = priority:
    later entries override earlier on name collision. Legacy
    ``skills_dir`` auto-migrated via model_validator (R5)."""

    scan_max_depth: int = 5
    """Maximum directory depth when scanning for SKILL.md files (R2).
    Paths deeper than this below a layer root are silently skipped.
    Prevents unbounded filesystem walks on huge mirrors."""

    # --- 检索/重排参数 ---
    embedding_model: str = "default"
    """Dense embedding model identifier. MUST match the embedding model
    that produced ``mass_library_db``'s stored vectors, otherwise dense
    retrieval returns garbage because the query vector lives in a different
    space. Configure this to match the embedding service and corpus used by
    your deployment."""

    embedding_url: str = "http://localhost:1357"
    """Remote embedding service base URL.

    Retrieval calls ``POST <embedding_url>/embed``. Override this with
    ``REMOTE_EMBEDDING_URL`` or user config when using a hosted embedding
    service."""

    reranker_enabled: bool = True
    """Run a reranker pass after dense retrieval. On by default — adds
    200-500ms per query (cross-encoder GPU inference) but lifts mass-pool
    precision noticeably. Disable when latency matters more than ranking."""

    reranker_model: str = "default"
    """Reranker model label used for configuration and observability."""

    reranker_url: str = "http://localhost:1357"
    """Remote reranker service base URL.

    Reranking calls ``POST <reranker_url>/score`` with
    ``{"prompts": [...]}`` and reads ``{"scores": [...]}``. Override this
    with ``REMOTE_RERANKER_URL`` or user config when using a hosted reranker
    service."""

    embedding_api_key: str | None = None
    """Optional bearer token for the configured embedding service."""

    reranker_api_key: str | None = None
    """Optional bearer token for the configured reranker service."""

    embedding_dimensions: int | None = None
    """Request specific embedding dimensions (for models that support it)."""

    top_k: int = 5
    """Number of skills returned by ``select()``."""

    # --- 双池融合权重（R6）---
    local_pool_top_k: int = 10
    """Candidate count from the local BM25 pool per query."""

    mass_pool_top_k: int = 10
    """Candidate count from the mass dense pool per query (post-rerank)."""

    local_weight: float = 1.3
    """RRF weight for local-pool candidates (mass is implicitly 1.0).
    Recommended range [1.2, 1.5]. Values < 1.0 or > 2.0 are rejected."""

    mass_reranker_overfetch: int = 20
    """When reranker is enabled, mass pool fetches this many candidates
    for rescoring, then truncates to ``mass_pool_top_k`` before RRF."""

    # --- 旧版查询改写参数 ---
    rewrite_enabled: bool = False
    """Legacy evaluation knob retained for config compatibility.

    The active Runtime never rewrites Local Skill queries with a Provider;
    Skill resolution is local-only so it cannot delay the main model call."""

    rewrite_max_tokens: int = 8192
    """Output token budget for the rewriter LLM call. Defaults to 8192 to
    leave headroom for Qwen3-style reasoning traces (~3-4k tokens) on top
    of the actual rewrite output. The previous 1024 budget caused frequent
    finish_reason=length truncations with empty visible content, which
    surfaced as 'Failed to parse rewrite response as JSON' fallbacks."""

    mass_library_db: str | None = None
    """Deprecated compatibility field ignored by Local Skill retrieval."""

    # --- 技能注入模式（full_body 或 summary）---
    injection_mode: str = "full_body"
    """How selected skills are surfaced to the agent.

    - ``"full_body"`` (default): inline the body of up to ``inject_max``
      explicit local matches and expose ambiguous matches as compact
      references.
    - ``"summary"``: expose all relevant matches as compact references.
      The Agent Loop calls ``skill_read`` to load a selected body."""

    inject_max: int = 2
    """Max skills inlined when ``injection_mode='full_body'``. Each skill body
    typically adds 1-5K tokens."""

    disable_always: bool = False
    """When True, ``get_always_skills()`` returns [] and select() filters
    out always:true skills. R8 default: False (always skills inject)."""

    always_max: int = 5
    """Max always skills injected per turn (R3). Exceeding this truncates
    by local_dirs list order + alphabetical, with a WARN listing dropped
    skill names."""

    # --- 旧版 LLM 门控选择器 ---
    llm_gate_enabled: bool = False
    """Legacy evaluation knob retained for config compatibility.

    The active Runtime uses deterministic local confidence instead of an LLM
    gate. Ambiguous candidates become compact references that the main Agent
    Loop can inspect with ``skill_read``."""

    llm_gate_max_select: int = 2
    """Upper bound on skills the gate may select. Mirrors ``inject_max``."""

    llm_gate_pool_size: int = 10
    """Candidate pool size handed to the gate (after RRF). Aligned
    with RRF output size (local_pool_top_k + mass_pool_top_k dedupe)."""

    llm_gate_model: str | None = None
    """Optional model override for gate calls. ``None`` → use the
    provider's default chat model (typically the agent's main model)."""

    llm_gate_temperature: float = 0.0
    """Sampling temperature for gate calls. 0.0 for deterministic
    filtering. Reasoning models may need 0.6 to engage <think>."""

    llm_gate_max_tokens: int = 8192
    """Output token budget for the gate LLM call. Defaults to 8192 to
    leave headroom for Qwen3-style reasoning traces (~3-4k tokens) on top
    of the gate's JSON answer. The previous 4096 budget caused empty
    content (finish_reason=length) on the 27B model in ~50% of calls,
    forcing a legacy top-N fallback that returned 5 skills instead of
    the configured llm_gate_max_select."""

    stats_tracking: bool = True
    """Record per-skill invocation stats. Cheap, enables future features."""

    # --- 校验器 ---

    @model_validator(mode="before")
    @classmethod
    def _migrate_skills_dir(cls, data: dict) -> dict:
        """R5: auto-convert legacy ``skills_dir`` → ``local_dirs``."""
        if not isinstance(data, dict):
            return data
        for old_key in ("skills_dir", "skillsDir"):
            old_val = data.pop(old_key, None)
            if old_val and "local_dirs" not in data and "localDirs" not in data:
                data["local_dirs"] = [{"path": old_val}]
                warnings.warn(
                    f"skill_forge.{old_key} is deprecated, use local_dirs "
                    f"instead. Auto-converted to local_dirs=[{{path: {old_val!r}}}]. "
                    f"This field will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        lw = data.get("local_weight") or data.get("localWeight")
        if lw is not None:
            lw = float(lw)
            if lw < 1.0 or lw > 2.0:
                raise ValueError(f"local_weight={lw} out of valid range [1.0, 2.0]")
        return data


# ---------------------------------------------------------------------------
# CFG-1：插件 / 记忆后端 / SkillForgeRouter
# ---------------------------------------------------------------------------


class PluginsConfig(_Base):
    """Plugin-system top-level config.

    ``disabled`` is the user opt-out list keyed by plugin id (matches
    the ``id`` in ``pico-plugin.toml``). ``config`` is the per-
    plugin config slice the registry hands to each plugin's factory
    via :class:`PluginContext.config` — its shape is determined by
    each plugin's own ``config_schema`` in the manifest, so the host
    treats it as a free-form dict.
    """

    disabled: list[str] = Field(default_factory=list)
    """Plugin ids the user opted out of."""

    config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-plugin configuration, keyed by plugin id. Each plugin's
    factory receives ``ctx.config = plugins.config.get(<id>, {})``."""


class MemoryConfig(_Base):
    """Which Memory backend is active and how recall is requested.

    ``backend`` is the name of an activated ``memory_backend``
    contribution. Set it to ``None`` to disable implicit Memory recall,
    persistence, personalization, and Curator Memory tools.

    ``user_id`` is the public Interface identity passed on the user recall
    track. Myna binds Memory by Workspace repository and does not use it as a
    repository namespace.
    """

    backend: str | None = "myna"
    """Activated backend contribution name. ``None`` disables the
    implicit Memory path while preserving Sessions, Curator state, and
    Local Skills."""

    user_id: str = "default"
    """Bare user identity passed as ``backend.recall(user_id=...)`` for
    the user-track recall channel inside ``ContextAssembler.assemble``."""

    memory_top_k: int = 5
    """Top-K passed to ``backend.recall(user_id=user_id)`` per turn for
    the ``# Recalled memory`` block."""


class SkillForgeRouterConfig(_Base):
    """Local Skill BM25 routing policy."""

    enabled: bool = True
    """Master switch. ``False`` makes the host bypass SkillForgeRouter
    entirely (used by tests / restricted deployments)."""

    local_min_score: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    """Minimum BM25 score emitted by the Local skill source."""

    @field_validator("local_min_score", mode="before")
    @classmethod
    def _reject_boolean_local_min_score(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("local_min_score must be a number")
        return value

    top_k: int = 5
    """Final top-K returned from ``SkillForgeRouter.select``."""


# ``SkillForgeRouterConfig`` 已存在于模块作用域，此处解析前向引用
# 字段声明：``SkillForgeConfig.router: "SkillForgeRouterConfig"``。
SkillForgeConfig.model_rebuild()


# ---------------------------------------------------------------------------
# 功能 4：Runtime 约束
# ---------------------------------------------------------------------------


class CheckpointConfig(_Base):
    """Per-turn shadow-git checkpoint of the workspace.

    When active, the agent loop commits the workspace to an out-of-band
    shadow git repo at the end of each turn (covering both normal and
    max-iteration exits). This is the safety net behind Bug2: a truncated
    multi-file edit leaves a recoverable snapshot, and the next turn gets a
    recovery prompt listing what the interrupted turn changed.

    Activation is gated by ``policy`` and the AgentLoop's ``interactive``
    flag (set per call site by the CLI / TUI / gateway entry points):

    - ``"always"``     — active in every AgentLoop, including ``-m``
                          one-shot commands.
    - ``"interactive"`` — active only when constructed for a multi-turn
                          session (REPL, TUI, gateway). One-shot commands
                          have no "next turn" to inject recovery into, so
                          paying the snapshot cost there is wasted.
    - ``"never"``      — disabled entirely; loop is byte-identical to the
                          pre-Bug2 baseline (no commits, no interrupt
                          reclassification, no recovery injection).

    Default ``"interactive"`` matches mature competitors (Claude Code,
    Cursor) which transparently checkpoint long sessions while leaving
    one-shot batch invocations untouched.
    """

    policy: Literal["always", "interactive", "never"] = "interactive"
    """When the per-turn shadow-git snapshot is active. See class
    docstring for the interaction with the AgentLoop ``interactive`` flag."""

    shadow_dir: str = ".pico/shadow.git"
    """Shadow git-dir, relative to Workspace State for project-local foreground
    runs and to the Workspace for colocated legacy/service runs. The real
    Workspace is the work-tree; the user's own ``.git`` is never touched."""


class RuntimeConfig(_Base):
    """Runtime discipline — the 4th feature pillar.

    Houses the opt-in runtime safety nets. Bug2 ships ``checkpoint``;
    later phases add ``journal`` / ``verifier`` / ``done_gate`` /
    ``loop_detection`` (Bug3, us) and ``session`` (Bug1, dev) as sibling
    sub-configs. All default off so the all-off baseline equals 68a3be7.
    """

    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class TracingConfig(_Base):
    """Observability tracing (in-tree ``pico.tracing``).

    On by default; every ``pico`` command auto-installs non-invasive
    instrumentation before any AgentLoop is built. ``PICO_TRACING=0`` is an
    explicit env kill-switch that overrides this block. View captured traces
    with ``pico tracing`` (or ``/tracing`` in the TUI).
    """

    enabled: bool = True
    port: int = 4318
    preview_len: int = 500


# ---------------------------------------------------------------------------
# 根配置
# ---------------------------------------------------------------------------


class PicoConfig(_Base):
    """Pico root config. Composes the base Config with feature extensions."""

    # 功能配置块
    context: ContextConfig = Field(default_factory=ContextConfig)
    call_efficiency: CallEfficiencyConfig = Field(default_factory=CallEfficiencyConfig)
    # SkillForge 子系统：其 RRF 路由策略嵌套在
    # ``skill_forge.router``（配置键 ``skillForge.router``），不再是
    # 而不是独立的顶层 ``skillRouter`` 块。
    skill_forge: SkillForgeConfig = Field(default_factory=SkillForgeConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)

    # CFG-1：插件系统与记忆后端。
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # 完整基础配置（agents、channels、providers、tools、routing）。
    # 保持为嵌套字段，以便与基础 loader 往返转换 YAML。
    base: BaseConfig = Field(default_factory=BaseConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_token_wise_block(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        legacy_camel = migrated.pop("tokenWise", None)
        legacy_snake = migrated.pop("token_wise", None)
        legacy = legacy_camel if legacy_camel is not None else legacy_snake
        if "callEfficiency" in migrated:
            migrated.pop("call_efficiency", None)
        if "callEfficiency" not in migrated and "call_efficiency" not in migrated and legacy is not None:
            migrated["callEfficiency"] = legacy
        return migrated

    @property
    def token_wise(self) -> CallEfficiencyConfig:
        """Compatibility view for historical code and frozen benchmarks."""
        return self.call_efficiency


def load_pico_config(config_path: Path | None = None) -> PicoConfig:
    """Load both the base Config and the Pico extension blocks
    (``context`` / ``call_efficiency`` / ``skill_forge``) from
    the same JSON config file.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Extension blocks fall through to their dataclass defaults when the
    JSON has no entry for them; explicit ``null`` values are also
    treated as "use default" rather than rejected.
    """
    base = load_base_config(config_path)

    overrides: dict = {}
    actual_path = config_path or get_config_path()
    if actual_path.exists():
        try:
            with open(actual_path, encoding="utf-8") as f:
                data = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            data = {}
        # 提取扩展块前，执行与基础 loader 相同的迁移。
        data = _migrate_config(data, pop_extension_keys=False)
        # 用户仍保留已忽略的旧配置时只警告一次。
        # ``skill_forge.mass_library_db`` 字段。
        _warn_mass_library_db_deprecated(data)
        for key in EXTENSION_KEYS:
            if key in data and data[key] is not None:
                overrides[key] = data[key]

    return PicoConfig(base=base, **overrides)


def _warn_mass_library_db_deprecated(data: dict) -> None:
    """Single-shot deprecation warning for ``skill_forge.mass_library_db``.

    The legacy SQLite field is retained for config compatibility but
    ignored by Local Skill retrieval.
    """
    legacy = None
    for skill_forge_key in ("skill_forge", "skillForge"):
        block = data.get(skill_forge_key)
        if isinstance(block, dict):
            legacy = block.get("mass_library_db") or block.get("massLibraryDb")
            if legacy:
                break
    if not legacy:
        return
    warnings.warn(
        "skill_forge.mass_library_db is deprecated and ignored. Local skills "
        "are discovered from configured filesystem sources; remove this field.",
        DeprecationWarning,
        stacklevel=2,
    )
