# Test Strategy

Use the canonical runner from the repository root:

```text
python scripts/run_tests.py fast --suite <suite>
python scripts/run_tests.py phase --phase <matrix-phase>
python scripts/run_tests.py global
```

The logical suites and their pytest targets are in
`scripts/testing/test_matrix.json`. Known platform and environment issues are
in `scripts/testing/known_environment_issues.json`.

## Tiers

**Fast** is the normal implementation loop. Select the changed feature's suite
or pass focused `--target` nodes. It runs one deduplicated pytest invocation,
Ruff on changed Python files, and `git diff --check`. It does not imply broad
historical regression, PicoBench, or collection coverage.

**Phase Acceptance** is used once when a subphase or phase is ready for human
commit/merge. It combines the phase suite, representative earlier-phase
regressions, relevant compatibility tests, and deterministic PicoBench tests in
one deduplicated pytest invocation. Do not concatenate prior Fast commands.

**Global Acceptance** is reserved for a major merge/release or an explicit
request. It preflights the whole deterministic test tree, excludes only exact
registered blockers/debt, runs the compatible remainder once, and reports
`BLOCKED` when the requested full contract could not be exercised.

Use `--dry-run` to inspect resolution and `--preflight-only` to inspect the
environment without running pytest. The runner uses a short
`.tmp/pytest/<tier>-<pid>` base temp on Windows before the first pytest launch.

## Results and retries

- `PASS` / exit 0: requested correctness checks passed with no blocking issue.
- `FAIL` / exit 1: a correctness, Ruff, diff, configuration, or unknown failure
  occurred.
- `BLOCKED` / exit 2: runnable checks passed, but a documented environment or
  platform blocker prevented the complete requested tier.

`FAIL` takes precedence over `BLOCKED`. Known test debt is never labelled as an
environment blocker. Exact stale nodes may be registered and deselected while
remaining tests in their file still run.

Correctness failures and unknown failures are never retried automatically.
Known missing dependencies or platform incompatibilities are not retried.
Path-length mitigation is configured before pytest. A transient external retry
is allowed only when its registry entry explicitly enables a bounded retry;
there are no such entries initially.

## Registering a new environment issue

On first occurrence, reproduce narrowly, rule out a code regression, record an
exact signature and affected targets, then add a registry entry and a cheap
preflight condition where possible. Update focused runner tests and this
document if handling changes. Never auto-register an unknown failure and never
keep retrying a broad suite. Future occurrences should cite the stable issue ID
and immediately apply its registered handling.

Add new tests to the narrowest reusable logical suite. Reuse suite references
instead of copying target lists; the runner reports requested targets, unique
targets, duplicates removed, and pytest process count.
