# Pico installation contract for agents

Use this document when an automation agent installs Pico for a user. Do not infer release URLs, expose secrets, publish artifacts, or initialize a repository the user did not select.

## Required inputs

- Target operating system: macOS/Linux or Windows.
- Absolute path of the target Git repository.
- Access to the private Gitee repository, or a trusted `PICO_WHEEL_URL`.
- Provider choice and credentials supplied directly by the user.
- Whether a real billed first Turn is allowed. Default to no without explicit authorization.

## Installation

For a private release, clone with the user's configured Gitee credentials and run the repository installer. Do not print credential values in logs.

```bash
git clone https://gitee.com/htxoffical/pico-harness.git
cd pico-harness
./install.sh
```

Windows PowerShell uses `install.ps1` from the same checkout.

## User-owned configuration boundary

Change into the exact target repository before onboarding:

```bash
cd <absolute-target-repository>
pico onboard --skip-memory --skip-test
```

The interactive wizard is the preferred secret-entry path. Only use `--non-interactive --api-key ...` when the user explicitly authorizes non-interactive secret handling; shell arguments can be visible to local process inspection and history tooling.

## Verification

Run read-only checks from the target repository:

```bash
uv tool list
pico plugins
pico channels list
pico doctor --json
```

Acceptance requires a valid `pico-harness` tool installation, valid doctor JSON, no unexpected Memory backend selection, and no credential value in captured output. Do not call a Provider verified unless an explicitly authorized live check passed.

## Feishu handoff

The user owns App ID/App Secret access, permission approval, application publication, and the inbound test message. Follow [feishu.zh-CN.md](feishu.zh-CN.md), then verify only redacted local state.
