"""Public Objective boundary, exact-point caching and feasible-point tracking."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


class BudgetExhausted(Exception):
    """Internal control flow for leaving SciPy as soon as the budget ends."""


class NumericalFailure(Exception):
    """A nonfinite trial, loss or gradient invalidates this local solve."""


@dataclass
class Diagnostics:
    """Work counters, not scores or the number of admitted history entries.

    Real evaluations include the call that exhausts the budget. Local solves
    count SciPy invocations; restarts count perturbed or global starting points.
    Solve exits partition those invocations on a completed run. Numerical
    warm-start failures are not solve exits. budget_exits counts overall
    budget termination, including exhaustion outside SciPy or a zero budget.
    """

    real_evaluations: int = 0
    cache_hits: int = 0
    warm_start_attempts: int = 0
    warm_adam_steps: int = 0
    local_solves: int = 0
    restarts: int = 0
    numerical_failures: int = 0
    solve_exits_converged: int = 0
    solve_exits_limited: int = 0
    solve_exits_abnormal: int = 0
    solve_exits_numerical: int = 0
    solve_exits_budget: int = 0
    budget_exits: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def as_vector(value, size: int, *, name: str, require_finite: bool = True) -> np.ndarray:
    """Accept real vectors only; own the storage even for readonly input arrays."""
    raw = np.asarray(value)
    if raw.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {raw.shape}")
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.array(raw, dtype=np.float64, copy=True)
    if require_finite and not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite float64 values")
    return vector


def _scalar_loss(value) -> float:
    raw = np.asarray(value)
    if raw.shape != ():
        raise ValueError("Objective loss must be scalar")
    if raw.dtype.kind not in "fiu":
        raise TypeError("Objective loss must be a real number")
    return float(raw)


def _scalar_feasibility(aux) -> bool:
    raw = np.asarray(aux["is_feasible"])
    if raw.shape != () or raw.dtype.kind != "b":
        raise ValueError("aux['is_feasible'] must be a scalar boolean")
    return bool(raw)


class EvaluationSession:
    """One logging run, with O(n_params) cache and best-point storage."""

    def __init__(self, objective, n_params: int, diagnostics: Diagnostics) -> None:
        self.objective = objective
        self.n_params = n_params
        self.diagnostics = diagnostics
        self._best_finite: tuple[float, np.ndarray] | None = None
        self._best_feasible: tuple[float, np.ndarray] | None = None
        self._cached: tuple[np.ndarray, float, np.ndarray] | None = None

    def check_budget(self) -> None:
        if self.objective.budget_exceeded:
            raise BudgetExhausted

    def clear_cache(self) -> None:
        self._cached = None

    def restart_center(self, fallback: np.ndarray) -> np.ndarray:
        best = self._best_feasible if self._best_feasible is not None else self._best_finite
        return fallback.copy() if best is None else best[1].copy()

    def _numerical_failure(self, message: str) -> None:
        self.diagnostics.numerical_failures += 1
        raise NumericalFailure(message)

    def __call__(self, params) -> tuple[float, np.ndarray]:
        self.check_budget()
        point = as_vector(params, self.n_params, name="trial parameters", require_finite=False)
        if not np.all(np.isfinite(point)):
            self._numerical_failure("Nonfinite solver trial parameters")
        if self._cached is not None and np.array_equal(point, self._cached[0]):
            gradient = self._cached[2].copy()
            self.check_budget()
            self.diagnostics.cache_hits += 1
            return self._cached[1], gradient

        self.clear_cache()
        # Objective may retain this array. It must never become solver or cache storage.
        evaluation_point = point.copy()
        self.check_budget()
        self.diagnostics.real_evaluations += 1
        loss_raw, gradient_raw, aux = self.objective.value_and_grad_aux(evaluation_point)
        loss = _scalar_loss(loss_raw)
        feasible = _scalar_feasibility(aux)
        if math.isfinite(loss):
            if self._best_finite is None or loss < self._best_finite[0]:
                self._best_finite = (loss, point.copy())
            if feasible and (self._best_feasible is None or loss < self._best_feasible[0]):
                self._best_feasible = (loss, point.copy())

        # A usable observation survives even when its derivative cannot drive a solve.
        gradient = as_vector(gradient_raw, self.n_params, name="Objective gradient", require_finite=False)
        if not math.isfinite(loss) or not np.all(np.isfinite(gradient)):
            self._numerical_failure("Nonfinite Objective loss or gradient")
        self._cached = (point, loss, gradient.copy())
        self.check_budget()
        return loss, gradient
