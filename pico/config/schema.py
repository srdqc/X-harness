"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings

from pico.product import DEFAULT_WORKSPACE_SPEC, get_default_workspace
from pico.sandbox.config import SandboxConfig


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class FeishuConfig(Base):
    """Feishu/Lark channel configuration using WebSocket long connection."""

    enabled: bool = False
    app_id: str = Field(default="", json_schema_extra={"required": True})  # 飞书开放平台 App ID
    app_secret: str = Field(default="", json_schema_extra={"required": True})  # 飞书开放平台 App Secret
    encrypt_key: str = ""  # 事件订阅 Encrypt Key
    verification_token: str = ""  # 事件订阅 Verification Token
    allow_from: list[str] = Field(default_factory=lambda: ["*"])  # 允许的用户 open_id；['*'] 表示任何人
    react_emoji: str = "THUMBSUP"  # 消息表态类型，如 THUMBSUP、OK、DONE、SMILE
    group_policy: Literal["open", "mention"] = "mention"  # "mention" 仅在被 @ 时响应，"open" 响应全部消息


class QQConfig(Base):
    """QQ channel configuration using botpy SDK."""

    enabled: bool = False
    app_id: str = Field(default="", json_schema_extra={"required": True})  # q.qq.com 机器人 AppID
    secret: str = Field(default="", json_schema_extra={"required": True})  # q.qq.com 机器人 AppSecret
    allow_from: list[str] = Field(default_factory=lambda: ["*"])  # 允许的用户 openid；['*'] 表示公开访问


class WecomConfig(Base):
    """WeCom (Enterprise WeChat) AI Bot channel configuration."""

    enabled: bool = False
    bot_id: str = Field(default="", json_schema_extra={"required": True})  # 企业微信 AI Bot 平台 Bot ID
    secret: str = Field(default="", json_schema_extra={"required": True})  # 企业微信 AI Bot 平台 Bot Secret
    allow_from: list[str] = Field(default_factory=lambda: ["*"])  # 允许的用户 ID；['*'] 表示任何人
    welcome_message: str = ""  # enter_chat 事件的欢迎消息


class ChannelsConfig(Base):
    """Configuration for chat channels."""

    send_progress: bool = True  # 向渠道流式发送 Agent 文本进度
    send_tool_hints: bool = False  # 流式发送工具调用提示，如 read_file("…")
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    qq: QQConfig = Field(default_factory=QQConfig)
    wecom: WecomConfig = Field(default_factory=WecomConfig)


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = DEFAULT_WORKSPACE_SPEC
    model: str = "anthropic/claude-opus-4-5"
    provider: str = "auto"  # Provider 名称，如 "anthropic"、"openrouter"；"auto" 表示自动检测
    max_tokens: int = 8192
    context_window_tokens: int = 65_536
    temperature: float = 0.1
    max_tool_iterations: int = 40
    # 同时运行的 subagent VM 上限；超额 spawn 会排队。ge=1：
    # 上限为 0 或负数会让所有子智能体死锁（Semaphore(0)）。
    max_concurrent_subagents: int = Field(default=4, ge=1)
    # 每个 session 在滚动一小时内的 spawn 速率限制。单靠并发门禁无法阻止
    # 被 prompt 注入的 Agent 无限 spawn：每个任务结束后都会释放槽位，跨 Turn
    # 重注入循环也不需要用户输入。滚动窗口把失控上限限制在 N 次/小时，且能
    # 自动恢复，不会永久阻止高强度的合法使用。按 session 分别计数，避免繁忙
    # session 限流其他 session。
    max_subagent_spawns_per_hour: int = Field(default=30, ge=1)
    # 空响应恢复：挽救模型未产生可见文本便结束、但实际上仍有内容可给出的 Turn，
    # 包括工具调用后为空或只有思考的情况，避免暴露无效的“无响应”结果。预算按 Turn 计算。
    empty_recovery_enabled: bool = True
    post_tool_empty_max_nudges: int = 1
    thinking_prefill_max_retries: int = 2
    empty_content_max_retries: int = 3
    # 已弃用的兼容字段：接受旧配置，但 Runtime 会忽略。
    memory_window: int | None = Field(default=None, exclude=True)
    reasoning_effort: str | None = None  # low / medium / high，用于启用 LLM thinking mode
    enable_personalization: bool = False  # 受 PAHF 启发的四步个性化流程：分类、询问、执行、学习

    @property
    def should_warn_deprecated_memory_window(self) -> bool:
        """Return True when old memoryWindow is present without contextWindowTokens."""
        return self.memory_window is not None and "context_window_tokens" not in self.model_fields_set


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class CronConfig(Base):
    """Cron scheduler configuration.

    Only consulted at cron job TRIGGER time, never at creation. Ephemeral
    channels (cli / tui — anything not in ChannelManager.enabled_channels)
    cannot deliver to themselves after the host process exits, so the
    forward_channels list resolves which real channels receive the reminder.
    """

    forward_channels: list[str] = Field(default_factory=lambda: ["*"])
    """Channels to deliver ephemeral-origin reminders to. ``["*"]`` broadcasts
    to every enabled channel. Specific names (``["feishu", "qq"]``)
    restrict to those. Non-ephemeral Channels ignore this list; they always
    pass through to the per-job Channel."""

    default_timezone: str = "Asia/Shanghai"
    """Default IANA timezone for cron expressions without explicit ``--tz``."""


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # 自定义 header，如 AiHubMix 的 APP-Code
    models: list[str] = Field(default_factory=list)  # 用户为选择器整理的模型名称


class GeminiProviderConfig(ProviderConfig):
    """Gemini provider configuration with Vertex AI and multi-key support.

    Example YAML:
        gemini:
          vertex: true
          api_key_list:
            - "key1"
            - "key2"
    """

    vertex: bool = False  # 为 True 时设置 GOOGLE_GENAI_USE_VERTEXAI=True，以使用 Vertex AI
    api_key_list: list[str] = Field(default_factory=list)  # 用于轮换的多个 API key

    def next_api_key(self) -> str:
        """Return the next API key using round-robin rotation.

        Falls back to single api_key if api_key_list is empty.
        """
        import itertools

        if not hasattr(self, "_key_cycle"):
            keys = self.api_key_list or ([self.api_key] if self.api_key else [])
            object.__setattr__(self, "_key_cycle", itertools.cycle(keys) if keys else None)
        cycle = getattr(self, "_key_cycle", None)
        if cycle is None:
            return self.api_key or ""
        return next(cycle)

    @property
    def effective_api_key(self) -> str:
        """Get the current effective API key (first from list, or single key)."""
        if self.api_key_list:
            return self.api_key_list[0]
        return self.api_key

    @property
    def all_keys(self) -> list[str]:
        """Return all configured API keys."""
        if self.api_key_list:
            return list(self.api_key_list)
        return [self.api_key] if self.api_key else []


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # 任意 OpenAI-compatible 端点
    azure_openai: ProviderConfig = Field(default_factory=ProviderConfig)  # Azure OpenAI，model 为部署名称
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)  # 阿里云通义千问
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: GeminiProviderConfig = Field(default_factory=GeminiProviderConfig)  # 提供商：Google Gemini / Vertex AI
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API 网关
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)  # Ollama 本地模型
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # 提供商：SiliconFlow
    volcengine: ProviderConfig = Field(default_factory=ProviderConfig)  # 提供商：VolcEngine
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig)  # 提供商：OpenAI Codex（OAuth）
    github_copilot: ProviderConfig = Field(default_factory=ProviderConfig)  # 提供商：Github Copilot（OAuth）


class ModelEndpoint(Base):
    """A routable model and the OpenAI-compatible endpoint that serves it."""

    model: str = ""
    api_base: str = ""
    api_key: str = "EMPTY"


class RoutingConfig(Base):
    """Model routing configuration.

    ``backend`` picks the router: ``ecoclaw`` (PinchBench benchmark scores, the
    original) or ``knn`` (task-level KNN over per-model rewards). Fields under
    "knn backend" are read only when ``backend == 'knn'``.
    """

    enabled: bool = False
    backend: str = "ecoclaw"  # 可选 ecoclaw 或 knn
    profile: str = "balanced"  # 可选 best、balanced 或 eco
    # 用于 embedding 的 OpenRouter API key（ecoclaw 后端；默认取 providers.openrouter.api_key）
    api_key: str = ""
    # knn 后端：可路由模型及其端点
    models: list[ModelEndpoint] = Field(default_factory=list)
    # knn 后端：预构建 KNN 记忆（embedding + 各模型奖励/成本）
    memory_path: str = ""
    k: int = 30  # 检索宽度：拉取多少个最近邻
    lambda_cost: float = 0.0  # 分数 = 奖励 - lambda_cost * 成本
    embedding_endpoint: str = ""  # 入站任务使用的 embedding 服务
    # knn 后端安全门禁：只有证据充分时才离开默认模型。
    # 在“相似”邻居（cosine >= min_similarity）上为候选模型评分。
    min_similarity: float = 0.6  # cosine 达到此值才将邻居视为相似
    min_similar_neighbors: int = 4  # 至少需要这么多相似邻居才路由
    min_memory_size: int = 10  # 至少需要这么多记忆条目才允许路由
    min_margin: float = 0.0  # 候选分数至少领先默认模型此值才切换


class GatewayLogConfig(Base):
    """Gateway logging configuration.

    ``rotation`` / ``retention`` accept loguru's vocabulary: rotation by size
    (``"10 MB"``), wall-clock (``"00:00"`` for daily), or interval
    (``"1 week"``); retention as a file count (``7``) or a duration
    (``"14 days"``).

    ``level`` filters the persisted ``gateway.log`` file; ``console_level``
    filters the live stderr mirror the foreground gateway keeps printing.
    """

    rotation: str = "10 MB"
    retention: int | str = 7
    level: str = "INFO"
    console_level: str = "INFO"


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790
    user_pool: int = 4
    system_pool: int = 2
    send_max_retries: int = 3
    log: GatewayLogConfig = Field(default_factory=GatewayLogConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    api_key: str = ""  # Serper API 密钥
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = None  # HTTP/SOCKS5 代理 URL，如 "http://127.0.0.1:7890" 或 "socks5://127.0.0.1:1080"
    jina_api_key: str = ""  # Jina Reader API 密钥
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    timeout: int = 60
    path_append: str = ""


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # 省略时自动检测
    command: str = ""  # Stdio：要运行的命令，如 "npx"
    args: list[str] = Field(default_factory=list)  # Stdio：命令参数
    env: dict[str, str] = Field(default_factory=dict)  # Stdio：额外环境变量
    url: str = ""  # HTTP/SSE：端点 URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE：自定义 header
    tool_timeout: int = 30  # 工具调用取消前等待的秒数


class ToolSearchConfig(Base):
    """Progressive tool disclosure.

    When the live tool catalog (built-ins + plugins + MCP) grows past
    ``compaction_threshold``, most tool schemas are withheld from each request and reached
    on demand through the ``tool_search`` / ``tool_call`` meta-tools, so context
    cost stops scaling with tool count and the per-turn tool list (and thus the
    prompt cache) stays stable. At or below the threshold every tool is exposed
    directly (unchanged behavior) and the meta-tools are omitted.
    """

    enabled: bool = False
    compaction_threshold: int = 50
    """Tool-catalog size that triggers compaction: at or below this many tools
    everything is exposed directly; above it, schemas are withheld."""
    search_result_limit: int = 10
    """Default number of hits ``tool_search`` returns per query."""
    always_visible: list[str] = Field(default_factory=list)
    """Extra tool names kept exposed every turn, on top of the core set."""


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # 为 True 时把所有工具访问限制在 workspace 目录
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    tool_search: ToolSearchConfig = Field(default_factory=ToolSearchConfig)
    disabled_tools: list[str] = Field(default_factory=list)
    """Tool names to unregister after default-tool registration and MCP connect.
    Used by eval harnesses (e.g. BrowseComp-Plus) that need to constrain the
    agent to a specific tool subset. Names match those in ``ToolRegistry``
    (e.g. ``read_file``, ``web_search``, or ``mcp_bcp-search_search``)."""


class Config(BaseSettings):
    """Root configuration for pico."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cron: CronConfig = Field(default_factory=CronConfig)
    # onboarding 时选择的 UI 语言，控制向导/CLI 文案和 Agent 回复语言
    # （注入 system prompt）。取值为 "en" | "zh"。
    language: Literal["en", "zh"] = "en"

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        workspace = self.agents.defaults.workspace
        return get_default_workspace() if workspace == DEFAULT_WORKSPACE_SPEC else Path(workspace).expanduser()

    def _match_provider(self, model: str | None = None) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name)."""
        from pico.providers.registry import PROVIDERS

        forced = self.agents.defaults.provider
        if forced != "auto":
            p = getattr(self.providers, forced, None)
            return (p, forced) if p else (None, None)

        model_lower = (model or self.agents.defaults.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # 显式 Provider 前缀优先，避免 `github-copilot/...codex` 匹配 openai_codex。
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and model_prefix and normalized_prefix == spec.name:
                if spec.is_oauth or spec.is_local or p.api_key:
                    return p, spec.name

        # 按关键字匹配，顺序与 PROVIDERS registry 一致。
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and any(_kw_matches(kw) for kw in spec.keywords):
                if spec.is_oauth or spec.is_local or p.api_key:
                    return p, spec.name

        # 回退：已配置的本地 Provider 可路由不含 Provider 专属关键字的模型，
        # 例如 Ollama 上的纯 "llama3.2"。
        for spec in PROVIDERS:
            if not spec.is_local:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.api_base:
                return p, spec.name

        # 回退：先网关，再按 registry 顺序尝试其他 Provider。
        # OAuth Provider 不能作为回退，必须显式选择模型。
        for spec in PROVIDERS:
            if spec.is_oauth:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.api_key:
                return p, spec.name
        return None, None

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers). Falls back to first available."""
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model)
        return p.api_key if p else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL for the given model. Applies default URLs for gateway/local providers."""
        from pico.providers.registry import find_by_name

        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        # 此处仅为网关设置默认 api_base。标准 Provider
        # Moonshot 等提供商会在 _setup_env 中通过环境变量设置 base URL。
        # 以避免污染全局 litellm.api_base。
        if name:
            spec = find_by_name(name)
            if spec and (spec.is_gateway or spec.is_local) and spec.default_api_base:
                return spec.default_api_base
        return None

    @property
    def skill_forge(self):
        """Returns the default SkillForgeConfig. Extension blocks are
        loaded via ``load_pico_config``, not through the base
        Config. This property exists for backward compat with code that
        accesses ``config.skill_forge`` on a plain ``Config`` instance.
        """
        from pico.config.pico import SkillForgeConfig

        return SkillForgeConfig()

    model_config = ConfigDict(
        env_prefix="PICO_",
        env_nested_delimiter="__",
        extra="forbid",
    )
