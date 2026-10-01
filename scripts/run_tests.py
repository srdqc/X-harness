#!/usr/bin/env python3
"""Canonical three-tier X-harness test entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.testing.runner import ConfigurationError, execute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=("fast", "phase", "global"))
    parser.add_argument("--suite", action="append", default=[], help="logical suite; may be repeated")
    parser.add_argument("--phase", help="phase key registered in the test matrix")
    parser.add_argument("--target", action="append", default=[], help="additional pytest node/path")
    parser.add_argument("--dry-run", action="store_true", help="resolve and report without running commands")
    parser.add_argument("--preflight-only", action="store_true", help="run preflight and report without commands")
    parser.add_argument("--skip-ruff", action="store_true", help="diagnostic use only")
    parser.add_argument("--skip-diff", action="store_true", help="diagnostic use only")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.tier == "phase" and not args.phase and not args.suite:
        build_parser().error("phase requires --phase or --suite")
    try:
        code, _summary = execute(
            tier=args.tier,
            suites=args.suite,
            phase=args.phase,
            extra_targets=args.target,
            dry_run=args.dry_run,
            preflight_only=args.preflight_only,
            skip_ruff=args.skip_ruff,
            skip_diff=args.skip_diff,
        )
    except ConfigurationError as exc:
        print(f"FAIL configuration: {exc}")
        return 1
    except KeyboardInterrupt:
        print('TEST_SUMMARY {"final":"FAIL","unknown_failures":["interrupted"]}')
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
