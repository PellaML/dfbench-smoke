"""Standard Adam settings and SciPy L-BFGS-B under the public Objective budget.

The warm phase uses the public AdamGD reference settings: clip 1.0, learning
rate 0.1 and default betas, with provenance in vendor/UPSTREAM.json. Local
solves use SciPy convergence and default iteration/evaluation safeguards.
A limit exit continues without noise; other completed exits allow restarts.
These are standard solver-policy choices, not a novelty or performance claim.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.optimize import minimize

from ._evaluation import BudgetExhausted, Diagnostics, EvaluationSession, NumericalFailure, as_vector

_WARM_ATTEMPTS = 12
_WARM_FRACTION = 0.15
_LOCAL_SCALES = (0.3, 0.8)
_LBFGS_OPTIONS = {"maxcor": 10, "maxls": 20, "gtol": 1e-6, "ftol": 1e-12}


def _integer(value, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _warm_attempt_limit(objective) -> int:
    max_evals, max_time = objective.max_evals, objective.max_time
    if max_evals is None and max_time is None:
        raise ValueError("HybridLBFGS requires a finite Objective evaluation or time budget")
    limit = _WARM_ATTEMPTS
    if max_evals is not None:
        max_evals = _integer(max_evals, name="max_evals")
        limit = min(limit, (3 * max_evals) // 20)
    if max_time is not None and (
        isinstance(max_time, (bool, np.bool_))
        or not isinstance(max_time, Real)
        or not np.isfinite(max_time)
        or max_time < 0
    ):
        raise ValueError("max_time must be a finite nonnegative number")
    return limit


def _clip_gradient(gradient: np.ndarray) -> np.ndarray:
    """Global L2 clipping without squaring or taking the norm of huge values."""
    largest = float(np.max(np.abs(gradient)))
    if largest == 0.0:
        return gradient.copy()
    scaled = gradient / largest
    scaled_norm = float(np.linalg.norm(scaled))
    if largest <= 1.0 / scaled_norm:
        return gradient.copy()
    return scaled / scaled_norm


def _warm_start(session: EvaluationSession, params: np.ndarray, attempts: int) -> np.ndarray:
    first = np.zeros_like(params)
    second = np.zeros_like(params)
    for step in range(1, attempts + 1):
        session.check_budget()
        # Wall time cannot be predicted or preempted within a single evaluation.
        if session.objective.budget_progress_fraction >= _WARM_FRACTION:
            break
        session.diagnostics.warm_start_attempts += 1
        _, gradient = session(params)
        gradient = _clip_gradient(gradient)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        corrected_first = first / (1.0 - 0.9**step)
        corrected_second = second / (1.0 - 0.999**step)
        updated = params - 0.1 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        session.diagnostics.warm_adam_steps += 1
        if np.array_equal(updated, params):
            break
        params = updated
    return params


def _restart(session: EvaluationSession, fallback: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    session.check_budget()
    phase = session.diagnostics.restarts % 3
    if phase == 2:
        point = session.objective.random_params()
    else:
        center = session.restart_center(fallback)
        point = center + rng.normal(0.0, _LOCAL_SCALES[phase], session.n_params)
    point = as_vector(point, session.n_params, name="restart parameters")
    session.diagnostics.restarts += 1
    return point


def _local_solve(session: EvaluationSession, params: np.ndarray) -> int | None:
    """Classify SciPy's L-BFGS-B exits; None denotes our numerical interruption."""
    session.check_budget()
    # Re-evaluate the start even after a limit exit so eval-only budgets advance.
    session.clear_cache()
    diagnostics = session.diagnostics
    evaluations_before = diagnostics.real_evaluations
    diagnostics.local_solves += 1
    try:
        result = minimize(
            session,
            params.copy(),
            method="L-BFGS-B",
            jac=True,
            callback=lambda _: session.check_budget(),
            options=_LBFGS_OPTIONS.copy(),
        )
    except NumericalFailure:
        diagnostics.solve_exits_numerical += 1
        return None
    except BudgetExhausted:
        diagnostics.solve_exits_budget += 1
        raise
    if diagnostics.real_evaluations == evaluations_before:
        raise RuntimeError("L-BFGS-B returned without an Objective evaluation")
    status = _integer(result.status, name="L-BFGS-B status")
    if status == 0:
        diagnostics.solve_exits_converged += 1
    elif status == 1:
        diagnostics.solve_exits_limited += 1
    elif status == 2:
        diagnostics.solve_exits_abnormal += 1
    else:
        raise ValueError(f"Unexpected L-BFGS-B status: {status}")
    return status


def run_hybrid(objective, init_params=None, *, random_seed: int) -> dict[str, int]:
    """Optimize an already-prepared unbounded Objective under its original budget.

    The caller resolves/seeds the Objective once. Only its matching aux warmup is
    unlogged. The warm phase has at most twelve attempts and 15% of the evaluation
    allowance; it stops starting attempts at 15% of the tightest budget's progress.
    An indivisible evaluation can cross the wall-time threshold. A finite public
    budget is required, since neither convergence nor numerical failure ends the
    overall search. SciPy limit exits continue at the best observed point with
    no perturbation; convergence, abnormal termination and numerical failure
    retain the local/local/global restart policy. No result.x is used.
    Diagnostics count calls and exit reasons, not performance or scores.
    """
    n_params = _integer(objective.n_params, name="n_params", minimum=1)
    seed = _integer(random_seed, name="random_seed")
    attempts = _warm_attempt_limit(objective)
    rng = np.random.default_rng(seed)
    params = as_vector(
        objective.random_params() if init_params is None else init_params,
        n_params,
        name="initial parameters",
    )
    diagnostics = Diagnostics()
    session = EvaluationSession(objective, n_params, diagnostics)
    objective.warmup_value_and_grad_aux()
    objective.start_logging()
    try:
        try:
            params = _warm_start(session, params, attempts)
        except NumericalFailure:
            params = _restart(session, params, rng)
        while True:
            status = _local_solve(session, params)
            session.check_budget()
            # A solver limit preserves observed progress without advancing the restart schedule.
            params = session.restart_center(params) if status == 1 else _restart(session, params, rng)
    except BudgetExhausted:
        diagnostics.budget_exits += 1
    return diagnostics.as_dict()
