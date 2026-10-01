"""Execute a reference optimizer and retain the organizer's result semantics."""

from __future__ import annotations

import platform
import time
from importlib.metadata import version
from typing import Any

from .config import ProbeConfig
from .vendor.validate_submission import summarize_objective

SOURCE_COMMIT = "64781a778ba546f83104c3692c343b16feb7eaf1"


def run_reference(objective: Any, optimizer: Any, seed: int) -> dict[str, Any]:
    """Use the reference method unchanged, including its own logging lifecycle."""
    started = time.monotonic()
    optimizer.optimize(objective, random_seed=seed)
    wall_seconds = time.monotonic() - started
    summary = summarize_objective(objective)
    return {
        "optimizer_wall_seconds": wall_seconds,
        "algorithm": str(optimizer.algorithm_str),
        **summary,
    }


def run_probe(config: ProbeConfig) -> dict[str, Any]:
    started = time.monotonic()
    if platform.system() != "Linux":
        raise RuntimeError("Simulator execution is supported in the Linux test container only.")

    # Keep scientific imports out of configuration checks and lightweight tests.
    import resource

    import jax
    from dfbench import Objective
    from dfbench.problems import ConstrainedVoyagerProblem, UIFOProblem

    if jax.default_backend() != "cpu":
        raise RuntimeError("This smoke test requires JAX_PLATFORMS=cpu.")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("This smoke test requires JAX_ENABLE_X64=true.")

    if config.method == "adam":
        from .vendor.adam_gd import AdamGD

        optimizer = AdamGD()
    else:
        from .vendor.random_search import RandomSearch

        optimizer = RandomSearch()

    problem = ConstrainedVoyagerProblem() if config.problem == "cvoyager" else UIFOProblem(topology_seed=config.seed)
    objective = Objective(
        problem,
        verbose=1,
        max_time=config.seconds,
        print_every=100,
        save=["is_feasible", "batched_loss", "batched_is_feasible"],
        display_mode="log",
        save_params_history=False,
        save_batched_params_history=False,
    )
    result = run_reference(objective, optimizer, config.seed)
    report = {
        "schema_version": 1,
        "problem": config.problem,
        "method": config.method,
        "optimizer_seed": config.seed,
        "topology_seed": config.seed if config.problem == "uifo" else None,
        "objective_budget_seconds": config.seconds,
        "parameter_count": int(objective.n_params),
        "source_commit": SOURCE_COMMIT,
        "python": platform.python_version(),
        "system": platform.system(),
        "backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "packages": {
            name: version(name) for name in ("dfbench", "differometor", "jax", "jaxlib", "optax", "numpy", "scipy")
        },
        # On Linux, ru_maxrss is KiB for this interpreter, not the Docker client.
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        **result,
        "scope": "One public CPU problem; not a competition or H100 score",
    }

    report["probe_wall_seconds"] = time.monotonic() - started
    return report
