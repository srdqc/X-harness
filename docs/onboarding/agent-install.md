# X-harness installation contract for agents

Use this contract when an automation agent installs X-harness for a user. The
installed distribution and CLI currently retain the compatible names
`pico-harness` and `pico`. The agent
must not infer release URLs, expose secrets, publish artifacts, reset existing
configuration, or initialize a repository the user did not select.

## Required inputs

- Target operating system: macOS/Linux or Windows.
- Absolute path of the target Git repository.
- A trusted checkout of this repository.
- Provider choice and credentials supplied directly by the user.
- Whether a real billed first Turn is allowed. Default to no without explicit
  authorization.
- Whether a message channel is in scope. Default to no.

The current release does not bundle an external Memory implementation. Use
`--skip-memory`; do not guess an unpublished repository, package, or download
address from adapter names in the source tree.

## Installation

Install from the trusted checkout:

```bash
uv sync --frozen --extra dev --dev
```

For an isolated command installation from the same checkout:

```bash
uv tool install --force ".[channels]"
```

Do not infer a package URL or install from an untrusted artifact. Never print,
persist, or copy provider and channel credentials into reports.

## User-owned configuration boundary

Change into the exact target repository before onboarding:

```bash
cd <absolute-target-repository>
pico onboard --skip-memory --skip-test
```

The interactive wizard is the preferred secret-entry path. Pause while the
user enters Provider credentials.

Only use `--non-interactive --api-key ...` when the user explicitly authorizes
non-interactive secret handling. Shell arguments can be visible to local
process inspection and history tooling.

Do not add `--reset` automatically. Existing Provider, channel, Sandbox, and
workspace choices belong to the user.

## Verification

Run read-only checks from the target repository:

```bash
uv tool list
pico --version
pico plugins
pico channels list
pico doctor --json
```

Acceptance requires:

- `uv tool list` contains `pico-harness` and exposes the `pico` executable.
- `pico --version` returns the installed version.
- `pico doctor --json` is valid JSON.
- Memory is explicitly disabled rather than pointed at an unavailable backend.
- No credential value appears in captured output.

If the user authorizes a billed live check, run one of these commands:

```bash
pico doctor --probe
pico run -m "Reply with: X-harness is ready"
```

Do not call a Provider verified unless the live command returned a model reply.
A successful installation, static doctor report, or skipped probe is not a live
Provider result.

## Feishu handoff

The user owns App ID and App Secret access, permission approval, application
publication, and the inbound test message. Follow
[feishu.zh-CN.md](feishu.zh-CN.md), then verify only redacted local state:

```bash
pico channels get feishu
pico gateway --workspace <absolute-target-repository> --verbose
```

Do not call Feishu connected until a human sends an inbound message and receives
the X-harness reply in the same conversation. Never paste Feishu secrets into an
issue, pull request, chat transcript, screenshot, or committed file.

## Handoff record

Report the installed X-harness version, target repository path, Memory state,
whether a billed probe ran, and whether a live channel round trip ran. Redact
credentials and signed URL query strings. If a layer was skipped, label it as
unverified instead of inferring success from another layer.
