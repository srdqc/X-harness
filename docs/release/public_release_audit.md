# Public Release Audit

Audit date: 2026-10-07  
Repository: X-harness  
Audited checkout: `test/jev6-v4-confirmatory` at `63f9083` plus the working-tree changes listed below

## 1. Executive Summary

**READY**

The proposed public-release scope is ready to be isolated into one release
commit. The public tree check, focused release tests, P0 phase acceptance, Ruff,
secret and privacy scans, and `git diff --check` pass after the fixes in this
working tree.

The working tree also contains pre-existing development changes. They are not a
release blocker because the release files can be staged by exact path. They must
not be included through `git add .` or `git add -A`.

The current branch is 23 commits ahead of `main`. A release commit created here
must be reviewed and integrated into `main` before the public release; the
feature branch itself must not be treated as the public release branch.

## 2. Repository State

Evidence from `git status --short`, `git diff --name-status`, and
`git diff --cached --name-status`:

- Current branch: `test/jev6-v4-confirmatory`.
- Divergence from `main`: 0 commits behind, 23 commits ahead.
- Staging area: empty.
- No commit, push, history rewrite, reset, clean, or rebase was performed by
  this audit.
- Tracked and untracked pre-existing development changes remain in place.

Classification:

### A. RELEASE_REQUIRED

- `.gitignore`
- `README.md`
- `README.zh-CN.md`
- `LICENSES/README.md`
- `NOTICES.md`
- `pyproject.toml`
- `benchmarks/README.md`
- `docs/evaluation/README.md`
- `docs/evaluation/runtime-scheduler-experiments.md`
- `docs/evaluation/tokenwise-cost.md`
- `docs/evaluation/tracing-overhead.md`
- `docs/onboarding/README.zh-CN.md`
- `docs/onboarding/agent-install.md`
- `docs/onboarding/feishu.zh-CN.md`
- `docs/onboarding/media-manifest.md`
- `docs/onboarding/memory.zh-CN.md`
- `docs/onboarding/troubleshooting.md`
- `docs/testing/TEST_STRATEGY.md` (deletion)
- `scripts/check_public_tree.py`
- `tests/test_public_release_tree.py`
- `benchmarks/picobench/packs/knowledge_evolution_live/jev6_v4_suite.py`
- `benchmarks/picobench/packs/knowledge_evolution_live/jev6_v4_frozen_manifest.json`
- `tests/test_jev6_v4_freeze.py`
- `docs/release/public_release_audit.md`

### B. RELEASE_RELATED

The tracked JEV6 V4 suite, manifest, and freeze test are release-related and are
included in `RELEASE_REQUIRED` because the committed versions contained a local
user-profile path. The manifest's suite digest was recomputed and its freeze
test passed after normalization to `~/.pico/config.json`.

### C. PRE_EXISTING_UNRELATED

These files were already modified before the public-release task and are
excluded from the release commit:

- `benchmarks/picobench/__init__.py`
- `benchmarks/picobench/packs/knowledge_evolution_live/__init__.py`
- `pico/knowledge_evolution/utility.py`
- `pico/providers/azure_openai_provider.py`
- `pico/providers/custom_provider.py`
- `pico/providers/openai_codex_provider.py`
- `tests/test_jev4_agent_pilot.py`
- `tests/test_litellm_provider_stream.py`
- `benchmarks/picobench/packs/knowledge_evolution_live/jev6_v2_benchmark.py`
- `tests/test_jev6_v2_confirmatory.py`

### D. UNCERTAIN

These untracked files existed before the release task and were subsequently
edited only to remove the same local path. Their full contents are pre-existing
feature work, so they are excluded from the release commit and must be reviewed
with that feature before any later commit:

- `benchmarks/picobench/packs/knowledge_evolution_live/jev6_v4_benchmark.py`
- `tests/test_jev6_v4_confirmatory.py`

## 3. README Accuracy Audit

The English and Chinese home pages were checked against the current CLI surface,
runtime boundaries, and deterministic tests.

- The public project name is X-harness.
- Pico is identified only as the original upstream project or in retained
  compatibility names.
- The current Python distribution and CLI remain `pico-harness` and `pico`;
  the README does not claim that these technical identifiers were renamed.
- The home-page order covers the problem, architecture, core features,
  execution flow, quick start, evaluation, and limitations/roadmap.
- Selective rewind, progressive disclosure, trace replay, knowledge evolution,
  and Jev remain marked **Experimental**.
- Gateway/channels and external Memory support remain marked **Partial** where
  live or external integration is not bundled or universally verified.
- Resume is described as a new Turn from durable evidence, not restoration of
  a coroutine, process, stream, lock, or program counter.
- No unsupported benchmark number is presented on the home page.
- Quick Start uses repository-local `uv` and lockfile commands that exist in the
  checkout; it does not depend on the former private-release instructions.

## 4. Sensitive Information Audit

The current tree was scanned with high-confidence patterns for common cloud,
GitHub, OpenAI-style, Slack, Google, private-key, and credential-bearing URL
formats.

Remaining matches were inspected without exposing their complete values. They
are deterministic test inputs, a private-key header test, or benchmark rule IDs:

- `tests/test_knowledge_scope.py`
- `tests/test_runtime_checkpoint_bug2_deep.py`
- `benchmarks/picobench/suites/agent_application_ship_1.yaml`
- `benchmarks/picobench/suites/agent_application_scorecard_v1.yaml`

No real credential was identified. Secret-bearing file extensions in the
release tree were not found.

## 5. Git History Audit

All refs were searched with the same high-confidence secret patterns. The only
historical matches were the same synthetic benchmark rule IDs and private-key
header test. No real secret was identified, so no history rewrite is required
or authorized.

## 6. Privacy / Local Environment Audit

The working tree was scanned for the known personal identifier and its Windows,
WSL, Linux, and macOS home-path forms. No match remains in the audited tree.

The committed JEV6 V4 provider source previously contained an absolute WSL user
profile path. It is normalized to `~/.pico/config.json`; the associated frozen
manifest digest and deterministic test were updated.

## 7. .gitignore Audit

The ignore rules cover:

- `.env`, `*.env`, and `key.env`;
- `*.key`, `*.pem`, `*.p12`, `*.pfx`, and `*.secret`;
- virtual environments, dependency directories, caches, build outputs, runtime
  state, worktrees, logs, and benchmark result directories.

No `.env.example` was added because the documented setup uses the interactive
configuration flow and repository-local source installation. Adding an example
without a stable environment-variable contract would be misleading.

## 8. Verification Results

Completed verification:

- Public tree check: **PASS** (`public release tree: OK`).
- Release-focused Fast run: **PASS**, 36 tests.
- P0 Phase Acceptance: **PASS**, 210 passed and 2 known-debt nodes deselected.
- Branding/public-tree follow-up Fast run: **PASS**, 23 tests.
- Final release-focused Fast run: **PASS**, 37 tests.
- Ruff through the canonical runner: **PASS**.
- `git diff --check`: **PASS**; Git reports only line-ending conversion warnings.
- Quick Start dependency command: **PASS**; the documented `uv sync` arguments
  plus `--dry-run` resolved the locked project environment without changing it.
- CLI entry point: **PASS**; `python -m pico --help` returned the current command
  surface.
- Personal/local environment scan: **PASS**, no remaining match.
- Secret-bearing path scan: **PASS**, no remaining path.
- High-confidence current-tree and history findings: inspected; no real secret.

The canonical preflight reports optional channel dependencies and historical
POSIX-only semantic tests as registered environment conditions. None affected
the requested release or P0 suites.

## 9. Known Limitations

- X-harness remains alpha; public interfaces may change before 1.0.
- Provider, sandbox, and channel behavior depends on local configuration and
  still requires environment-specific live smoke tests.
- External Memory is not bundled.
- Experimental and Partial capabilities retain those labels; this audit does
  not promote their maturity.
- The distribution name and CLI remain `pico-harness` and `pico` for current
  compatibility. A full package/CLI rename is not part of this release audit.
- The current branch contains 23 commits not yet in `main`; integration must be
  reviewed before public push.
- Untracked V4 confirmatory files are mixed pre-existing work and are excluded
  from this release commit.

`docs/testing/TEST_STRATEGY.md` is deleted because the public-tree contract
allows onboarding, examples, evaluation evidence, and this release audit, but
not internal development-process documentation. Repository-wide search found no
README or public-document reference to it, so the deletion creates no broken
public link. The canonical runner and its machine-readable test matrix remain
the executable source of truth.

## 10. Release Scope

Only the files in section 2.A should enter the release commit. Exact-path
staging is required. Do not stage section 2.C or 2.D.

The release commit can be isolated safely despite the dirty worktree. After the
commit, review whether the 23 prerequisite branch commits belong in the public
release. Integrate the reviewed release state into `main`, then rerun the public
tree check and release tests on `main` before pushing.

## 11. Final Release Checklist

- [x] README matches the current implementation and project identity.
- [x] Experimental, Partial, and Roadmap status is explicit.
- [x] Public tree check passes.
- [x] Current-tree secret scan completed; findings are non-secret fixtures.
- [x] Git history secret scan completed; no real secret found.
- [x] Personal and local-environment paths removed from release scope.
- [x] `.gitignore` covers common local credentials and generated state.
- [x] Release-focused tests pass.
- [x] P0 Phase Acceptance passes with registered deselections only.
- [x] Ruff passes.
- [x] `git diff --check` passes.
- [x] Release files can be staged by exact path.
- [ ] Release commit created and reviewed.
- [ ] Reviewed release commit integrated into `main`.
- [ ] Verification rerun on `main` immediately before public push.

`PUBLIC_PUSH_READY=YES`

This value means the audited, exact-path release scope is ready for commit and
integration. It does not authorize pushing the current dirty feature branch.
