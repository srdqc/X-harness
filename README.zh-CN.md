# Pico Agent Harness

Pico 是一个面向终端、自动化任务与消息渠道的 Agent Harness。它把一次请求组织成可调度、可取消、可追踪、可验证的 Turn，并提供 CLI、TUI、Gateway、Tool/MCP、Session、Tracing 与 PicoBench。

当前仓库是 Pico 的发布仓库。它只保留可发布源码、确定性测试、公开 benchmark fixture、构建与安装文件、onboarding 文档和法律文件；开发过程文档、原始运行数据、真实凭证与私有环境说明不在此仓库发布。

## 环境要求

- macOS、Linux 或 Windows
- Python 3.12
- Node.js 22 或更高版本（安装器可下载私有运行时）

## 安装

仓库处于私有阶段时，先使用已配置的 Gitee 凭证克隆，再运行安装器：

```bash
git clone https://gitee.com/htxoffical/pico-harness.git
cd pico-harness
./install.sh
```

Windows PowerShell：

```powershell
git clone https://gitee.com/htxoffical/pico-harness.git
Set-Location pico-harness
.\install.ps1
```

安装器默认从 Gitee Release 获取 Pico wheel，使用清华 PyPI 镜像和阿里云 Node.js 镜像。私有 Release 需要设置 `PICO_GITEE_TOKEN`；也可以用 `PICO_WHEEL_URL` 指定经过信任的 wheel。

```bash
export PICO_GITEE_TOKEN="<your-gitee-token>"
printf 'Authorization: Bearer %s\n' "$PICO_GITEE_TOKEN" | \
  curl -fsSL -H @- https://gitee.com/htxoffical/pico-harness/raw/main/install.sh | sh
```

PowerShell：

```powershell
$headers = @{ Authorization = "Bearer $env:PICO_GITEE_TOKEN" }
irm https://gitee.com/htxoffical/pico-harness/raw/main/install.ps1 -Headers $headers | iex
```

可用环境变量：

- `PICO_WHEEL_URL`：直接指定 Pico wheel
- `PICO_GITEE_TOKEN`：读取私有 Gitee Release
- `PICO_PYPI_INDEX`：覆盖 Python 包索引
- `PICO_NODE_MIRROR`：覆盖 Node.js 镜像
- `PICO_NODE_CHECKSUM_BASE`：覆盖 Node.js 校验清单来源；默认使用上游官方地址
- `PICO_NPM_REGISTRY`：覆盖 npm registry
- `PICO_UV_INSTALL_URL`：覆盖 uv 安装脚本地址

## 第一次运行

当前发布不包含外部 Memory 实现，首次配置请跳过 Memory：

```bash
pico onboard --skip-memory
pico
pico run -m "你好，Pico"
```

更多内容见 [onboarding](docs/onboarding/README.zh-CN.md)。

## 开发与验证

```bash
uv sync --frozen --extra dev --dev
npm ci
npm ci --prefix ui-tui
make check
make picobench-smoke
PICO_RELEASE_OUTPUT=/absolute/empty/output make release-dist
```

PicoBench 的公开任务、fixture 与 verifier 位于 [`benchmarks/`](benchmarks/)。仓库中的结果只代表对应文档写明的冻结实验，不能外推为线上 SLA 或未运行环境的性能结论。

## 许可证

Pico 使用 Apache License 2.0。第三方归属与许可证见 [NOTICES.md](NOTICES.md) 和 [LICENSES/](LICENSES/)。
