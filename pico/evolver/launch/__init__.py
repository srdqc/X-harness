"""Unified launch layer: one config file + one command for any registered bench.

``pico evolve run --config <yaml>`` drives the whole SOP flow as a
resumable state machine: cold-start thick ledger -> evolution rounds ->
terminate -> unseal. Interrupt anywhere; re-running the same command resumes
from the last durable artifact (trial files / round journal / meta stamps).
"""

from pico.evolver.launch.contract import BenchBundle, LaunchContext, validate_whitelist
from pico.evolver.launch.registry import load_bench

__all__ = ["BenchBundle", "LaunchContext", "load_bench", "validate_whitelist"]
