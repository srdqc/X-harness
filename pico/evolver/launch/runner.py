"""The run/status/finalize state machine over durable artifacts.

Resume model: work is proven by artifacts, not by process state.

- phase 1 (cold start): proof = vanilla trial result files. Resume fills
  only the missing trials.
- phase 2 (rounds): proof = journal/rounds.jsonl. Resume replays completed
  rounds (loop.run's built-in journal replay) and continues.
- phase 3 (unseal): proof = the ``unsealed_at`` stamp in run_meta.json.
  Unsealing is one-way — a stamped run refuses to resume (--force overrides,
  marking any further rounds as retention-invalid is on the caller).

``status`` never reads the sealed directory: while a run is resumable, test
numbers must stay invisible (SOP §0) — the only path to them is natural
termination or an explicit ``finalize``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from pico.evolver.activation import write_evolution_summary
from pico.evolver.launch.config import RunSpec, RunSpecError, load_run_spec
from pico.evolver.launch.contract import BenchBundle, LaunchContext
from pico.evolver.launch.models import build_role_call_fns, describe_models
from pico.evolver.launch.registry import load_bench
from pico.evolver.launch.state import META_FILENAME, RunMeta, atomic_write_json, load_json_or
from pico.evolver.tree import git_ops
from pico.utils.portable_lock import LockTimeoutError, file_lock


def _say(msg: str) -> None:
    print(f"[evolve] {msg}", flush=True)


def _load_spec(config_path: str, smoke: bool) -> RunSpec:
    try:
        return load_run_spec(config_path, smoke=smoke)
    except RunSpecError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _build_bundle(
    spec: RunSpec,
    *,
    with_models: bool,
    require_sealed: bool = False,
) -> BenchBundle:
    models = {
        "driver": None,
        "design": None,
        "verdict": None,
    }
    try:
        build = load_bench(spec.bench, repo_root=spec.repo_root)
        bundle = build(LaunchContext(spec=spec, models=models))
    except ValueError as exc:
        print(f"bench setup error ({spec.bench}): {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if require_sealed and bundle.unseal is None:
        print(
            f"bench setup error ({spec.bench}): a sealed test is required for "
            "readiness; configure a non-overlapping test split",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if with_models:
        try:
            models.update(build_role_call_fns(spec.models))
        except ValueError as exc:
            print(f"models error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    return bundle


def _note_defaulted_base(spec: RunSpec) -> None:
    if not spec.base_sha_defaulted:
        return
    _say(
        f"base_sha not set — using repo HEAD {spec.base_sha[:12]} "
        "(pin base_sha in the yaml to freeze the root explicitly)"
    )
    proc = subprocess.run(
        ["git", "-C", str(spec.repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    if proc.stdout.strip():
        _say(
            "warning: repo_root has uncommitted changes — they are NOT part "
            "of the root node (evaluations check out commits, not the "
            "working tree)"
        )


def _evolution_lock_path(spec: RunSpec) -> Path:
    return spec.work_dir.parent / f".{spec.work_dir.name}.evolution.lock"


def _remove_registered_worktrees(spec: RunSpec) -> None:
    work_dir = spec.work_dir.resolve()
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(spec.repo_root),
            "worktree",
            "list",
            "--porcelain",
            "-z",
        ],
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        print(
            f"cannot inspect registered worktrees before cleanup: {listed.stderr.strip()}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    for field in listed.stdout.split("\0"):
        if not field.startswith("worktree "):
            continue
        target = Path(field.removeprefix("worktree ")).resolve()
        if target != work_dir and work_dir not in target.parents:
            continue
        removed = subprocess.run(
            [
                "git",
                "-C",
                str(spec.repo_root),
                "worktree",
                "remove",
                "--force",
                str(target),
            ],
            capture_output=True,
            text=True,
        )
        if removed.returncode != 0 and target.exists():
            print(
                f"cannot remove stale Evolution Run worktree {target}: {removed.stderr.strip()}",
                file=sys.stderr,
            )
            raise SystemExit(2)


def _claim_ephemeral_root(spec: RunSpec, meta: RunMeta) -> None:
    """Park ephemeral git worktrees under work_dir; sweep hard-kill leftovers.

    Normal exits (including Ctrl-C) clean them via context managers — only a
    SIGKILL'd run leaves them behind, still registered against the subject
    repo. A valid persisted RunMeta proves this is a dedicated run directory
    before any prior ``tmp`` tree is removed.
    """
    persisted = RunMeta.load(spec.work_dir)
    marker = Path(spec.work_dir) / META_FILENAME
    if (
        persisted is None
        or marker.is_symlink()
        or persisted.config_hash != meta.config_hash
        or persisted.created_at != meta.created_at
    ):
        print(
            f"refusing worktree cleanup: {spec.work_dir} is not owned by the active Evolution Run",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _remove_registered_worktrees(spec)
    tmp_root = Path(spec.work_dir) / "tmp"
    for stale_root in (tmp_root, Path(spec.work_dir) / "wt"):
        if stale_root.is_symlink():
            print(f"refusing worktree cleanup through symlink: {stale_root}", file=sys.stderr)
            raise SystemExit(2)
        if stale_root.exists():
            shutil.rmtree(stale_root)
    subprocess.run(
        ["git", "-C", str(spec.repo_root), "worktree", "prune"],
        capture_output=True,
        text=True,
    )
    git_ops.set_ephemeral_root(tmp_root)


def _meta_guard(spec: RunSpec, *, force: bool) -> RunMeta:
    """Config-drift + one-way-unseal guards; returns the (possibly new) meta."""
    snapshot = spec.snapshot()
    meta = RunMeta.load(spec.work_dir)
    if meta is None:
        if any(Path(spec.work_dir).iterdir()):
            print(
                f"refusing to claim nonempty work_dir without a valid {META_FILENAME}: {spec.work_dir}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return RunMeta.create(spec.work_dir, snapshot)
    if meta.unsealed_at and not force:
        print(
            f"this run was unsealed at {meta.unsealed_at} "
            f"(reason: {meta.finalize_reason}); resuming would leak test "
            "numbers into decisions. Start a fresh work_dir, or pass --force "
            "to continue with retention marked invalid.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not meta.check_config(snapshot) and not force:
        print(
            "config drift: the effective configuration differs from the one "
            "this work_dir was started with (same-regime rule, SOP §0). "
            "Candidate and control arms would no longer be comparable. "
            "Use a fresh work_dir for the new config, or --force to override.",
            file=sys.stderr,
        )
        if spec.base_sha_defaulted:
            print(
                "note: base_sha is omitted in the yaml, so it re-resolved to "
                f"the current HEAD ({spec.base_sha[:12]}); if the repo gained "
                "commits since the run started, pin base_sha to the original "
                f"root ({meta.config_snapshot.get('base_sha', '?')[:12]}) to "
                "resume.",
                file=sys.stderr,
            )
        raise SystemExit(2)
    return meta


def _run_rounds(bundle: BenchBundle):
    from pico.evolver.orchestrator.state.journal import RoundJournal

    orch = bundle.build_orchestrator()
    journal = RoundJournal(bundle.journal_path)
    result = orch.run(bundle.root_node_id, journal=journal, root_node=bundle.root_node)
    return orch, journal, result


def _refresh_summary(spec: RunSpec) -> Path:
    return write_evolution_summary(spec.work_dir)


def _unseal_and_report(
    spec: RunSpec, bundle: BenchBundle, orch, records: list[dict], meta: RunMeta, reason: str
) -> bool:
    """Returns True on success; False when unseal scoring failed (not stamped)."""
    if bundle.unseal is None:
        _say("no sealed test set configured; skipping unseal")
        if not meta.unsealed_at:
            meta.stamp_unsealed(reason=f"{reason} (no sealed test)")
        return True
    _say("unsealing: blind-scoring deliverables on the sealed test set …")
    try:
        report = bundle.unseal(records, orch)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"unseal scoring failed: {exc}", file=sys.stderr)
        print(
            "the run is NOT stamped; fix the environment and re-run the same command to retry the unseal",
            file=sys.stderr,
        )
        return False
    # 写报告前先盖章：该标记负责阻止恢复，因此不能存在崩溃窗口，让测试数字留在仍可恢复的
    # 运行目录中。盖章后崩溃会留下缺少 retention.json 的最终运行；`finalize` 会检测并重算。
    if not meta.unsealed_at:
        meta.stamp_unsealed(reason=reason)
    atomic_write_json(Path(spec.work_dir) / "retention.json", report)
    _say(f"retention report -> {Path(spec.work_dir) / 'retention.json'}")
    for key in (
        "best_round",
        "best_node_id",
        "best_train",
        "best_test",
        "vanilla_train",
        "vanilla_test",
        "retention",
        "sealed_credited_2sigma",
    ):
        if isinstance(report, dict) and key in report:
            _say(f"  {key}: {report[key]}")
    return True


def cmd_run(config_path: str, *, smoke: bool = False, force: bool = False) -> int:
    spec = _load_spec(config_path, smoke)
    try:
        with file_lock(_evolution_lock_path(spec), blocking=False):
            return _cmd_run(spec, force=force)
    except LockTimeoutError:
        print(
            f"another mutating Evolution Run process already owns {spec.work_dir}",
            file=sys.stderr,
        )
        return 2


def _cmd_run(spec: RunSpec, *, force: bool) -> int:
    _note_defaulted_base(spec)

    # 创建 run_meta 前构建并校验 bundle：首次启动若因配置错误失败，不得留下指纹，否则下次
    # 尝试时修正后的配置会被误判为漂移并拒绝。
    bundle = _build_bundle(spec, with_models=True, require_sealed=True)

    spec.work_dir.mkdir(parents=True, exist_ok=True)
    meta = _meta_guard(spec, force=force)
    _claim_ephemeral_root(spec, meta)
    meta.config_snapshot.setdefault("resolved_models", describe_models(spec.models))
    meta.save()

    done, total = bundle.cold_start_done(), bundle.cold_start_total
    if done < total:
        _say(f"phase 1/3 cold start: {done}/{total} trials present, running the rest …")
    else:
        _say(f"phase 1/3 cold start: {total} trials present, verifying the infra-rerun ladder …")
    # 始终调用：run_cold_start 具备幂等性，只填补缺失试验；它还负责基础设施重跑阶梯，即使
    # 所有基础试验文件都存在，也可能仍有抢救工作要做。
    try:
        bundle.run_cold_start()
    except KeyboardInterrupt:
        _refresh_summary(spec)
        _say(
            f"interrupted during cold start "
            f"({bundle.cold_start_done()}/{total} trials done and kept); "
            "re-run the same command to continue"
        )
        return 130
    except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
        print(f"cold start failed: {exc}", file=sys.stderr)
        print(
            f"completed trials are kept "
            f"({bundle.cold_start_done()}/{total} present); fix the "
            "environment and re-run the same command to resume",
            file=sys.stderr,
        )
        return 1
    done = bundle.cold_start_done()
    if done < total:
        _say(f"cold start still incomplete ({done}/{total}); re-run to retry the missing trials")
        return 1
    _say(f"phase 1/3 cold start complete ({total} trials)")

    _say("phase 2/3 evolution rounds (interrupt any time; same command resumes)")
    _say(
        f"  live progress: {spec.work_dir}/findings.md (per-round log), "
        f"{spec.work_dir}/journal/rounds.jsonl (checkpoints)"
    )
    try:
        orch, journal, result = _run_rounds(bundle)
    except KeyboardInterrupt:
        _refresh_summary(spec)
        _say(
            "interrupted — completed rounds are journaled; "
            "re-run the same command to resume, `status` to inspect, "
            "`finalize` to stop here and unseal"
        )
        return 130
    except RuntimeError as exc:
        _refresh_summary(spec)
        # 环境类失败（门控 0 预检、端点失效）应显示可执行消息而非回溯；已完成工作保持持久。
        print(f"run stopped: {exc}", file=sys.stderr)
        print("fix the environment and re-run the same command to resume", file=sys.stderr)
        return 1

    for rr in result.rounds:
        _say(f"round {rr.round_index}: parent={rr.parent_id} promoted={rr.promoted} -> {rr.next_parent_id}")
    _say(f"stopped: {result.stop_reason}; final parent: {result.final_parent_id}")

    _say("phase 3/3 unseal")
    ok = _unseal_and_report(spec, bundle, orch, journal.load(), meta, reason=result.stop_reason or "terminated")
    _refresh_summary(spec)
    return 0 if ok else 1


def cmd_check(config_path: str, *, smoke: bool = False) -> int:
    """Validate everything cheap before spending: config, models, bench setup.

    Builds the model call_fns (catches a missing claude binary / bad spec) and
    the bench bundle (catches dead whitelist entries, missing task files or
    subject config, absent AppWorld install) without running anything.
    """
    spec = _load_spec(config_path, smoke)
    _note_defaulted_base(spec)
    bundle = _build_bundle(spec, with_models=True, require_sealed=True)
    _say(f"bench:    {spec.bench} (root {spec.base_sha[:12]} @ {spec.repo_root})")
    _say(f"work_dir: {spec.work_dir}")
    _say(f"models:   {describe_models(spec.models)}")
    _say(
        f"funnel:   k_screen={spec.funnel.k_screen} k_confirm={spec.funnel.k_confirm} "
        f"budget={spec.funnel.budget.max_why_per_round}x"
        f"{spec.funnel.budget.candidates_per_why} "
        f"rounds<={spec.funnel.termination.max_rounds}"
    )
    _say(f"cold start: {bundle.cold_start_done()}/{bundle.cold_start_total} trials present")
    _say(f"sealed test: {'configured' if bundle.unseal else 'not configured'}")
    if bundle.precheck is not None:
        _say("bench precheck: probing environment + subject endpoint …")
        try:
            bundle.precheck()
        except RuntimeError as exc:
            print(f"bench precheck failed: {exc}", file=sys.stderr)
            return 1
        _say("bench precheck: OK")
    _say("check OK — ready to run")
    return 0


def _node_status_counts(work_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(Path(work_dir).glob("nodes/*.json")):
        rec = load_json_or(path, {})
        status = rec.get("status", "?")
        counts[status] = counts.get(status, 0) + 1
    return counts


def cmd_status(config_path: str, *, smoke: bool = False) -> int:
    spec = _load_spec(config_path, smoke)
    meta = RunMeta.load(spec.work_dir)
    if meta is None:
        _say(f"no run state under {spec.work_dir} (phase 0: not started)")
        return 0
    _say(f"work_dir: {spec.work_dir}")
    _say(f"started: {meta.created_at}  config: {meta.config_hash}")
    summary_path = _refresh_summary(spec)
    summary = load_json_or(summary_path, {})
    counts = summary.get("outcome_counts", {}) if isinstance(summary, dict) else {}
    if counts:
        _say(
            "evidence outcomes: "
            + ", ".join(f"{name}={counts.get(name, 0)}" for name in ("accepted", "rejected", "failed", "inconclusive"))
        )
    integrity_error_count = summary.get("integrity_error_count", 0) if isinstance(summary, dict) else 0
    if integrity_error_count:
        _say(f"evidence integrity errors: {integrity_error_count} (affected candidates are classified failed)")
    _say(f"deterministic summary: {summary_path}")
    if meta.unsealed_at:
        _say(f"UNSEALED at {meta.unsealed_at} ({meta.finalize_reason}) — run is final")
        report = load_json_or(Path(spec.work_dir) / "retention.json", None)
        if report:
            _say(f"retention report: {Path(spec.work_dir) / 'retention.json'}")
        return 0

    bundle = _build_bundle(spec, with_models=False)
    done, total = bundle.cold_start_done(), bundle.cold_start_total
    if done < total:
        _say(f"phase 1: cold start {done}/{total} trials — no results yet")
        return 0

    from pico.evolver.orchestrator.state.journal import RoundJournal

    records = RoundJournal(bundle.journal_path).load()
    if not records:
        _say("phase 2: cold start done, no completed rounds yet")
        return 0
    _say(f"phase 2: {len(records)} completed round(s); test stays sealed until termination or `finalize`")
    counts = _node_status_counts(spec.work_dir)
    if counts:
        _say("candidates by status: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    promoted = []
    for rec in records:
        _say(
            f"  round {rec.get('round_index')}: promoted={rec.get('promoted')} "
            f"beat_vanilla={rec.get('beat_vanilla')} "
            f"parent -> {rec.get('next_parent_id')}"
        )
        if rec.get("promoted") and rec.get("next_parent_sha"):
            promoted.append((rec.get("next_parent_id"), rec.get("next_parent_sha"), rec.get("next_parent_train")))
    if promoted:
        _say("promoted commits (train-side numbers only):")
        for node_id, sha, train in promoted:
            _say(f"  {node_id} @ {sha}  train={train}")
    return 0


def cmd_finalize(config_path: str, *, smoke: bool = False, yes: bool = False) -> int:
    spec = _load_spec(config_path, smoke)
    try:
        with file_lock(_evolution_lock_path(spec), blocking=False):
            return _cmd_finalize(spec, yes=yes)
    except LockTimeoutError:
        print(
            f"another mutating Evolution Run process already owns {spec.work_dir}",
            file=sys.stderr,
        )
        return 2


def _cmd_finalize(spec: RunSpec, *, yes: bool) -> int:
    meta = RunMeta.load(spec.work_dir)
    if meta is None:
        print("nothing to finalize: run was never started", file=sys.stderr)
        return 2
    if not meta.check_config(spec.snapshot()):
        print(
            "config drift: finalize requires the same effective configuration that owns this Evolution Run",
            file=sys.stderr,
        )
        return 2
    recompute = bool(meta.unsealed_at)
    if recompute and (Path(spec.work_dir) / "retention.json").is_file():
        _refresh_summary(spec)
        _say(f"already unsealed at {meta.unsealed_at}; see retention.json")
        return 0

    bundle = _build_bundle(spec, with_models=False)
    if recompute:
        if bundle.unseal is None:
            # 未执行密封测试便已最终化，因此没有报告可重建。
            _say(f"already finalized at {meta.unsealed_at} ({meta.finalize_reason}); no sealed test was configured")
            return 0
        _say(
            "unseal stamp present but retention.json is missing (interrupted "
            "unseal); recomputing the report — the run stays final"
        )
    from pico.evolver.orchestrator.state.journal import RoundJournal

    records = RoundJournal(bundle.journal_path).load()
    if not records:
        print("nothing to finalize: no completed rounds in the journal", file=sys.stderr)
        return 2
    if not yes and not recompute:
        what = (
            "and unseal the test set"
            if bundle.unseal is not None
            else "(no sealed test configured — no test numbers exist)"
        )
        print(
            f"finalize will END this run after {len(records)} round(s) {what} "
            "— it cannot be resumed afterwards. Re-run with --yes.",
            file=sys.stderr,
        )
        return 2

    # 解封会在测试集上给节点评分：不需要编排器及其构建路径所需模型，但需要基准评测，bundle
    # 闭包已携带它。vanilla_train 来自构建后的编排器，因此在无 LLM 角色的情况下构建；
    # 这些角色只在轮次中调用。
    _claim_ephemeral_root(spec, meta)
    orch = bundle.build_orchestrator()
    ok = _unseal_and_report(spec, bundle, orch, records, meta, reason="user_finalized")
    _refresh_summary(spec)
    return 0 if ok else 1


__all__ = ["cmd_run", "cmd_status", "cmd_check", "cmd_finalize"]
