# Pico Agent Harness

Pico is an Agent Harness for terminal workflows, automated tasks, and message channels. It turns each request into a schedulable, cancellable, traceable, and verifiable Turn, with CLI, TUI, Gateway, Tool/MCP, Session, Tracing, and PicoBench surfaces.

This is the release repository. It contains publishable source code, deterministic tests, public benchmark fixtures, build and installer files, onboarding material, and legal notices. Development notes, raw run artifacts, credentials, and private-environment instructions are intentionally excluded.

## Requirements

- macOS, Linux, or Windows
- Python 3.12
- Node.js 22 or newer; the installer can provision a private runtime

## Install

While the repository is private, clone it with your configured Gitee credentials and run the local installer:

```bash
git clone https://gitee.com/htxoffical/pico-harness.git
cd pico-harness
./install.sh
```

Windows PowerShell:

```powershell
git clone https://gitee.com/htxoffical/pico-harness.git
Set-Location pico-harness
.\install.ps1
```

The installer resolves Pico wheels from Gitee Releases and defaults to China-hosted Python and Node.js mirrors. Set `PICO_GITEE_TOKEN` for a private Release, or set `PICO_WHEEL_URL` to a trusted wheel URL.

## First run

This release does not bundle an external Memory implementation. Configure Pico without Memory:

```bash
pico onboard --skip-memory
pico
pico run -m "hello"
```

See the [Chinese onboarding guide](docs/onboarding/README.zh-CN.md) for the supported setup path.

## Development and verification

```bash
uv sync --frozen --extra dev --dev
npm ci
npm ci --prefix ui-tui
make check
make picobench-smoke
PICO_RELEASE_OUTPUT=/absolute/empty/output make release-dist
```

Public benchmark tasks, fixtures, and verifiers live under [`benchmarks/`](benchmarks/). Reported measurements apply only to their documented frozen experiments; they are not production SLAs.

## License

Pico is licensed under Apache License 2.0. Third-party notices and license texts are in [NOTICES.md](NOTICES.md) and [LICENSES/](LICENSES/).
