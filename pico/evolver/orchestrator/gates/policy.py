"""Pluggable decision policy — the seam the two benchmark lines diverge on.

The seven-step funnel's outer loop is identical across benchmarks; only the
per-candidate *decision* (screen -> confirm -> promote) and the *control arm* it
compares against differ. Those two concerns are captured here as two protocols
so ``loop._run_round`` stays a thin driver:

- :class:`GatePolicy` — given a :class:`DecisionContext`, run whatever
  screen/confirm/significance logic the bench uses and return one
  :class:`CandidateOutcome`. The policy owns its own ``eval`` calls because the
  two lines eval different task sets at different stages (K=1 anchor vs K=3
  focused subset). The loop never scores for itself anymore.
- :class:`BaselineProvider` — supply the control arm (a :class:`Baseline`) for
  the round and absorb the ratchet on promotion. Frozen cold-start, per-parent
  frozen, and same-session paired are the three implementations; the gate never
  knows which produced its control, which is why the two seams are orthogonal.

The empirical regime-shift finding (a frozen baseline compared across time is
unreliable when a whole run can shift ~12pp) is why
:class:`SameSessionPairedBaseline` exists and is the methodology-correct choice;
the frozen variants are kept as cost-bound fallbacks and labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from pico.evolver.orchestrator.gates.fisher import train_mean
from pico.evolver.orchestrator.scoring import EvalFn, EvaluationVerdict, TaskEval
from pico.evolver.scheduler.anchor_selection import AnchorSelection
from pico.evolver.tree.node import HarnessNode, NodeStatus

if TYPE_CHECKING:
    from pico.evolver.orchestrator.gates.paired import PairedResult
    from pico.evolver.orchestrator.gates.pipeline import GateResult
    from pico.evolver.orchestrator.nodes.screen import ScreenResult


# （节点，任务 ID）-> 已触发任务集合；节点无归因数据（候选项未插桩或未接入收集）时为 None。
# None 让门控 b 开放通过（跳过），空集合则如实表示“从未触发”。
FiredSourceFn = Callable[[HarnessNode, list[str]], Optional[set]]
FocusedSourceFn = Callable[[HarnessNode], list[str]]


@dataclass
class CandidateOutcome:
    """What happened to one candidate as it moved through the funnel.

    ``score``/``confirm_evals``/``stats`` are policy-agnostic so parent
    selection and the baseline ratchet stay in the loop, not in the policies.
    The optional ``screen``/``paired``/``gate`` are the paired-line detail;
    ``stats`` carries the Fisher-line detail (fisher_p, foc_c, full_lift, ...).
    """

    node_id: str
    status: NodeStatus
    score: float = 0.0
    confirm_evals: dict[str, TaskEval] = field(default_factory=dict)
    screen: Optional["ScreenResult"] = None
    paired: Optional["PairedResult"] = None
    gate: Optional["GateResult"] = None
    stats: dict = field(default_factory=dict)
    verdict: EvaluationVerdict | None = None

    def __post_init__(self) -> None:
        if isinstance(self.verdict, str):
            self.verdict = EvaluationVerdict(self.verdict)
        if self.verdict is not None:
            return
        if self.status is NodeStatus.promoted_to_baseline:
            self.verdict = EvaluationVerdict.accepted
        elif self.status is NodeStatus.errored:
            self.verdict = EvaluationVerdict.failed
        else:
            self.verdict = EvaluationVerdict.rejected

    @property
    def promoted(self) -> bool:
        return self.status == NodeStatus.promoted_to_baseline and self.verdict is EvaluationVerdict.accepted


@dataclass(frozen=True)
class Baseline:
    """The control arm for a round: per-task evals + their train mean + a label."""

    evals: dict[str, TaskEval]
    mean: float
    label: str


@dataclass
class DecisionContext:
    """Everything the loop hands a :class:`GatePolicy` to decide one candidate."""

    node: HarnessNode
    parent_id: str
    round_index: int
    eval: EvalFn
    baseline: Baseline
    train_task_ids: list[str]
    anchor: Optional[AnchorSelection] = None
    focused_task_ids: list[str] = field(default_factory=list)
    # 作为退化防护带入聚焦评测的稳定通过对照任务（SOP 第 2 节第 5a 项，“2 到 3 个全通过
    # 哨兵”）：帮助 WHY 子集的修复不得破坏原本通过的任务。空集合表示不防护。
    sentinel_task_ids: list[str] = field(default_factory=list)
    fired_source: Optional[FiredSourceFn] = None


class GatePolicy(Protocol):
    def decide(self, ctx: DecisionContext) -> CandidateOutcome: ...


class BaselineProvider(Protocol):
    def for_round(
        self,
        round_index: int,
        parent: HarnessNode,
        *,
        eval: EvalFn,
        train_task_ids: list[str],
        anchor: Optional[AnchorSelection],
    ) -> Baseline: ...

    def on_promote(self, node: HarnessNode, outcome: CandidateOutcome, *, train_task_ids: list[str]) -> None: ...


class FrozenColdStartBaseline:
    """One vanilla cold-start control, reused every round (SWE default).

    Cross-time-invalid (compares later rounds against a run that may be in a
    different regime); a cost-bound fallback, not the methodology-correct choice.
    """

    def __init__(self, control_evals: dict[str, TaskEval], *, label: str = "vanilla"):
        self._evals = dict(control_evals)
        self._label = label

    def for_round(self, round_index, parent, *, eval, train_task_ids, anchor) -> Baseline:
        return Baseline(self._evals, train_mean(self._evals, train_task_ids), self._label)

    def on_promote(self, node, outcome, *, train_task_ids) -> None:  # 冻结基线永不移动
        return None


class PerParentFrozenBaseline:
    """Per-parent frozen control; a promoted node's confirm evals become its
    children's baseline (the AppWorld ratchet). Also cross-time-invalid.

    ``fallback(parent, train_task_ids)`` rebuilds a missing parent's baseline
    from durable artifacts (e.g. its confirm out-dir on disk) — the resume path:
    after a process restart only the seed survives in memory, but a promoted
    parent's confirm evals are still on disk. Return None when the artifacts are
    gone; ``for_round`` then raises rather than silently comparing to nothing.
    """

    def __init__(
        self,
        seed: dict[str, Baseline],
        *,
        fallback: Optional[Callable[[HarnessNode, list[str]], Optional[Baseline]]] = None,
    ):
        self._by_parent: dict[str, Baseline] = dict(seed)
        self._fallback = fallback

    def for_round(self, round_index, parent, *, eval, train_task_ids, anchor) -> Baseline:
        baseline = self._by_parent.get(parent.node_id)
        if baseline is None and self._fallback is not None:
            baseline = self._fallback(parent, list(train_task_ids))
            if baseline is not None:
                self._by_parent[parent.node_id] = baseline
        if baseline is None:
            raise KeyError(
                f"no baseline for parent {parent.node_id!r} "
                f"(resumed without a fallback, or its confirm artifacts are gone)"
            )
        return baseline

    def on_promote(self, node, outcome, *, train_task_ids) -> None:
        # 完整训练集均值（SOP 第 0 节固定分母）：确认运行遗漏的任务仍必须按 0 计，使该基线与
        # 下一轮候选组使用相同分母基础。
        evals = outcome.confirm_evals
        mean = train_mean(evals, train_task_ids)
        self._by_parent[node.node_id] = Baseline(evals, mean, f"{node.node_id}_confirm")


class SameSessionPairedBaseline:
    """Re-run the parent's own harness this round as the control (vanilla at C0).

    Methodology-correct under regime shift — candidate and control are always
    measured in the same session/window — at ~2x eval cost. Requires ``eval`` to
    reproduce the parent node's harness (the backend's job).
    """

    def __init__(self, k: int, *, label: str = "control"):
        self._k = k
        self._label = label

    def for_round(self, round_index, parent, *, eval, train_task_ids, anchor) -> Baseline:
        evals = eval(parent, train_task_ids, self._k, f"{self._label}_r{round_index}")
        return Baseline(evals, train_mean(evals, train_task_ids), f"{parent.node_id}_{self._label}_r{round_index}")

    def on_promote(self, node, outcome, *, train_task_ids) -> None:  # 每轮重新测量
        return None


def make_frozen_baseline(
    *,
    root_node_id: str,
    vanilla_dir: "Path",
    kept_reader: Callable[["Path"], dict[str, TaskEval]],
    confirm_dir_of: Callable[[HarnessNode], "Path"],
    train_task_ids: list[str],
    seed_label: str,
) -> PerParentFrozenBaseline:
    """Assemble a :class:`PerParentFrozenBaseline` seeded from a vanilla ledger.

    The bench-neutral shape both benches share: the seed baseline is the vanilla
    ledger read through ``kept_reader`` (the infra-rerun KEPT overlay, so the
    control arm sees the same salvage rule candidate evals get — SOP §0); the
    resume fallback re-reads a promoted parent's confirm dir (``confirm_dir_of``)
    the same way. The only bench-specific inputs are the reader (out-dir vs
    job-dir format) and the confirm-dir locator; everything else is identical.
    Must be called after the vanilla cold start has materialised the ledger.
    """
    van_evals = kept_reader(vanilla_dir)

    def fallback(parent: HarnessNode, train_ids: list[str]) -> Optional[Baseline]:
        d = vanilla_dir if parent.node_id == root_node_id else confirm_dir_of(parent)
        try:
            evals = kept_reader(d)
        except FileNotFoundError:
            return None
        return Baseline(evals, train_mean(evals, train_ids), f"{d.name}(resumed)")

    return PerParentFrozenBaseline(
        seed={root_node_id: Baseline(van_evals, train_mean(van_evals, list(train_task_ids)), seed_label)},
        fallback=fallback,
    )


__all__ = [
    "CandidateOutcome",
    "Baseline",
    "DecisionContext",
    "GatePolicy",
    "BaselineProvider",
    "FrozenColdStartBaseline",
    "PerParentFrozenBaseline",
    "SameSessionPairedBaseline",
    "make_frozen_baseline",
    "FiredSourceFn",
    "FocusedSourceFn",
]
