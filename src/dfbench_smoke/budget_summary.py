"""Summarize only observations whose public timestamps fit the time budget."""

from __future__ import annotations

import math
from itertools import pairwise
from numbers import Real
from types import SimpleNamespace
from typing import Any

from .vendor.validate_submission import summarize_objective

_HISTORY_FIELDS = (
    "loss_history",
    "is_feasible_history",
)


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "time_window_status": "unavailable",
        "time_window_reason": reason,
        "time_window_record_count": None,
        "after_budget_record_count": None,
        "time_window_finite_candidate_count": None,
        "time_window_feasible_candidate_count": None,
        "time_window_missing_feasibility_candidate_count": None,
        "time_window_best_feasible_loss": None,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def summarize_time_window(objective: Any) -> dict[str, Any]:
    """Keep the organizer's scalar/batch rules, but exclude late observations.

    This is a diagnostic view over recorded data, not a replacement scorer.
    A missing clock cannot establish a within-budget result.
    """
    budget = _number(getattr(objective, "max_time", None))
    if budget is None or budget <= 0:
        return _unavailable("no_positive_finite_time_budget")
    losses = getattr(objective, "loss_history", None)
    timestamps = getattr(objective, "time_steps", None)
    if losses is None or timestamps is None or len(losses) != len(timestamps):
        return _unavailable("missing_or_misaligned_timestamps")
    if not len(losses) and getattr(objective, "eval_count", 0):
        return _unavailable("missing_loss_history")

    times = [_number(value) for value in timestamps]
    if any(value is None or value < 0 for value in times):
        return _unavailable("invalid_timestamps")
    if any(current < previous for previous, current in pairwise(times)):
        return _unavailable("nonmonotonic_timestamps")
    # Follow scoring.md's closed [0, T] interval. These are the Objective's
    # recorded wall-clock readings, not independent completion timestamps.
    retained = [index for index, elapsed in enumerate(times) if elapsed <= budget]

    histories = {name: getattr(objective, name, None) for name in _HISTORY_FIELDS}
    histories["loss_history"] = losses
    # Padding keeps short aux histories aligned instead of assigning a later
    # flag to an earlier candidate after filtering.
    selected = {
        name: [history[index] if history is not None and index < len(history) else None for index in retained]
        for name, history in histories.items()
    }
    summary = summarize_objective(SimpleNamespace(**selected))
    return {
        "time_window_status": "complete",
        "time_window_reason": None,
        "time_window_record_count": len(retained),
        "after_budget_record_count": len(losses) - len(retained),
        "time_window_finite_candidate_count": summary["finite_candidate_count"],
        "time_window_feasible_candidate_count": summary["feasible_candidate_count"],
        "time_window_missing_feasibility_candidate_count": summary["missing_feasibility_candidate_count"],
        "time_window_best_feasible_loss": summary["best_feasible_loss"],
    }
