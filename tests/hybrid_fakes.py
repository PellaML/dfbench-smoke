"""Analytic objectives and deterministic clocks for optimizer tests."""

from types import SimpleNamespace

import numpy as np


def quadratic(params):
    target = np.arange(1.0, params.size + 1.0)
    residual = params - target
    return float(residual @ residual), 2.0 * residual, True


def flat(params):
    return 1.0, np.zeros_like(params), True


class AnalyticObjective:
    """A small public-API fake; time advances only on real evaluations."""

    def __init__(self, function=quadratic, *, size=2, max_evals=80, max_time=None, eval_seconds=0.01, seed=13):
        self.function = function
        self.n_params = size
        self.max_evals = max_evals
        self.max_time = max_time
        self.eval_seconds = eval_seconds
        self.elapsed_extra = 0.0
        self.eval_count = 0
        self.started = False
        self.warmed = False
        self.events = []
        self.points = []
        self.retained_points = []
        self.losses = []
        self.feasibility = []
        self.sample_count = 0
        self.sample = None
        self.rng = np.random.default_rng(seed)

    @property
    def time_elapsed(self):
        return self.elapsed_extra + self.eval_count * self.eval_seconds

    @property
    def budget_exceeded(self):
        if not self.started:
            raise AssertionError("Budget check before logging started")
        return (self.max_evals is not None and self.eval_count >= self.max_evals) or (
            self.max_time is not None and self.time_elapsed >= self.max_time
        )

    @property
    def budget_progress_fraction(self):
        if not self.started:
            raise AssertionError("Progress read before logging started")
        fractions = [0.0]
        if self.max_evals is not None:
            fractions.append(self.eval_count / self.max_evals if self.max_evals else 1.0)
        if self.max_time is not None:
            fractions.append(self.time_elapsed / self.max_time if self.max_time else 1.0)
        return min(1.0, max(fractions))

    def random_params(self):
        if self.started and self.budget_exceeded:
            raise AssertionError("Random sample requested after budget exhaustion")
        self.events.append("sample")
        self.sample_count += 1
        if self.sample is not None:
            return self.sample
        return self.rng.uniform(-3.0, 3.0, self.n_params)

    def warmup_value_and_grad_aux(self):
        if self.started or self.warmed:
            raise RuntimeError("Warmup must run once before logging")
        self.events.append("warmup_aux")
        self.warmed = True

    def start_logging(self):
        if self.started or not self.warmed:
            raise RuntimeError("Logging must start once after warmup")
        self.events.append("start_logging")
        self.started = True

    def value_and_grad_aux(self, params):
        if not self.started or not self.warmed or self.budget_exceeded:
            raise AssertionError("Evaluation outside the one logging budget")
        self.events.append("evaluate_aux")
        self.points.append(params.copy())
        self.retained_points.append(params)
        loss, gradient, feasible = self.function(params)
        self.eval_count += 1
        self.losses.append(loss)
        self.feasibility.append(feasible)
        return loss, gradient, {"is_feasible": np.asarray(feasible)}


def one_evaluation_solve(fun, params, **kwargs):
    fun(params)
    return SimpleNamespace(x=np.full_like(params, np.nan), fun=-1e99, success=False, status=2)
