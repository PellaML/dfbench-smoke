"""Command-line entry point for bounded reference runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .config import ProbeConfig
from .probe import run_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short CPU test on a full public dfbench problem.")
    parser.add_argument("--problem", choices=("cvoyager", "uifo"), required=True)
    parser.add_argument("--method", choices=("adam", "random"), default="adam")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-feasible", action="store_true", help="exit 3 unless a finite feasible result was logged"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = ProbeConfig(args.problem, args.method, args.seconds, args.seed)
    except ValueError as error:
        parser.error(str(error))
    try:
        result = run_probe(config)
        payload = json.dumps(result, sort_keys=True, allow_nan=False)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"dfbench-smoke: {error}", file=sys.stderr)
        return 1
    print(f"DFBENCH_RESULT={payload}", flush=True)
    if args.require_feasible and not result["feasible_candidate_count"]:
        print("dfbench-smoke: no finite physically feasible result was recorded", file=sys.stderr)
        return 3
    return 0
