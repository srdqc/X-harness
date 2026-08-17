# Pico 首次使用

这份指南只做一件事：让你在一个 Git 仓库里收到 Pico 的第一条真实回复。

## 安装

Pico 需要 Python 3.12。原生 TUI 需要 Node.js 22，安装脚本会在系统缺失时下载私有 Node runtime。

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

不要从来路不明的地址安装 wheel。使用私有 Gitee Release 时设置 `PICO_GITEE_TOKEN`；需要固定制品时设置 `PICO_WHEEL_URL`。

## 在目标仓库中完成向导

```bash
cd /path/to/your-project
pico onboard --skip-memory
```

向导按用户能看见的结果排序：连接 Provider、发送第一条真实 Turn、选择运行位置，再决定是否接入消息渠道。当前发布不包含外部 Memory 实现，因此 onboarding 的受支持路径是显式跳过 Memory。

自动化场景使用：

```bash
pico onboard \
  --non-interactive \
  --provider openai \
  --api-key "$OPENAI_API_KEY" \
  --skip-memory \
  --skip-channel \
  --yes
```

`--skip-test` 会跳过第一条付费 Turn。这种情况下，Provider 配置成功不等于模型已经真实回复。

## 验收

```bash
pico doctor --probe
pico run -m "用三句话说明这个仓库做什么"
```

需要结构化结果时：

```bash
pico doctor --json
```

## 继续配置

- [飞书机器人](feishu.zh-CN.md)
- [Memory 发布边界](memory.zh-CN.md)
- [故障排查](troubleshooting.md)
- [给安装 Agent 的操作合同](agent-install.md)
