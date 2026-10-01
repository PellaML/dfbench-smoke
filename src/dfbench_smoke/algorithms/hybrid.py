"""Thin dfbench adapter for the NumPy/SciPy hybrid candidate."""

from __future__ import annotations

from dfbench import OptimizationAlgorithm

from .core import run_hybrid


class HybridLBFGS(OptimizationAlgorithm):
    """Clipped Adam initialization followed by budgeted, restarted L-BFGS."""

    algorithm_str = "hybrid_lbfgs"

    def __init__(self) -> None:
        self.diagnostics: dict[str, int] = {}

    def optimize(self, objective, init_params=None, random_seed=None) -> None:
        self.diagnostics = {}
        resolved_seed, _ = self.prepare(objective, unbounded=True, random_seed=random_seed)
        self.diagnostics = run_hybrid(objective, init_params=init_params, random_seed=resolved_seed)
