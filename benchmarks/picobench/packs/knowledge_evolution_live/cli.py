"""Explicit human CLI for preparing, running, and reducing P3R campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmarks.picobench.canonical import canonical_json

from .artifacts import load_manifest, load_runs, store_for
from .prepare import preflight, prepare_campaign
from .reducer import reduce_campaign
from .runner import execute_one, run_campaign
from .schema import CampaignMode, CampaignPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.picobench.packs.knowledge_evolution_live")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="freeze and preflight a campaign without Provider calls")
    prepare.add_argument("--repository", type=Path, default=Path.cwd())
    prepare.add_argument("--output-root", type=Path, default=Path(".p3r"))
    prepare.add_argument("--campaign", choices=[item.value for item in CampaignMode], required=True)
    prepare.add_argument("--base-commit", required=True)
    prepare.add_argument("--campaign-seed", type=int, default=31_415)
    prepare.add_argument("--reviewer-id", required=True)

    check = commands.add_parser("preflight", help="recheck a frozen campaign without Provider calls")
    check.add_argument("--repository", type=Path, default=Path.cwd())
    check.add_argument("--campaign-root", type=Path, required=True)
    check.add_argument("--reviewer-id", required=True)

    run = commands.add_parser("run", help="execute only missing frozen live runs")
    run.add_argument("--repository", type=Path, default=Path.cwd())
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--reviewer-id", required=True)
    run.add_argument("--execute-live", action="store_true")

    reduce = commands.add_parser("reduce", help="reduce immutable records offline")
    reduce.add_argument("--campaign-root", type=Path, required=True)

    internal = commands.add_parser("_run-one")
    internal.add_argument("--repository", type=Path, required=True)
    internal.add_argument("--campaign-root", type=Path, required=True)
    internal.add_argument("--run-id", required=True)
    internal.add_argument("--reviewer-id", required=True)
    internal.add_argument("--execute-live", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from pico.config.loader import load_config

        paths, manifest, result = prepare_campaign(
            repository=args.repository.resolve(),
            output_root=args.output_root.resolve(),
            mode=CampaignMode(args.campaign),
            base_commit=args.base_commit,
            seed=args.campaign_seed,
            reviewer_id=args.reviewer_id,
            config=load_config(),
        )
        print(
            canonical_json(
                {
                    "campaign_id": manifest.campaign_id,
                    "campaign_root": paths.root,
                    "planned_live_runs": len(manifest.planned_runs),
                    "preflight": result,
                    "live_provider_invoked": False,
                }
            )
        )
        return 0
    if args.command == "preflight":
        from pico.config.loader import load_config

        paths = CampaignPaths.at(args.campaign_root.resolve())
        manifest = load_manifest(paths.manifest)
        result = preflight(
            repository=args.repository.resolve(),
            base_commit_sha=manifest.base_commit_sha,
            reviewer_id=args.reviewer_id,
            config=load_config(),
            manifest=manifest,
        )
        print(canonical_json({"preflight": result, "live_provider_invoked": False}))
        return 0
    if args.command == "run":
        run_campaign(
            repository=args.repository.resolve(),
            campaign_root=args.campaign_root.resolve(),
            reviewer_id=args.reviewer_id,
            execute_live=args.execute_live,
        )
        return 0
    if args.command == "_run-one":
        execute_one(
            repository=args.repository.resolve(),
            campaign_root=args.campaign_root.resolve(),
            run_id=args.run_id,
            reviewer_id=args.reviewer_id,
            execute_live=args.execute_live,
        )
        return 0
    paths = CampaignPaths.at(args.campaign_root.resolve())
    result = reduce_campaign(load_manifest(paths.manifest), load_runs(paths))
    store_for(paths).write_summary(paths.reduced, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]
