import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from dfbench_smoke.budget_summary import summarize_time_window


def objective(losses, feasible, times, budget=10.0, **extra):
    return SimpleNamespace(
        loss_history=losses,
        is_feasible_history=feasible,
        time_steps=times,
        max_time=budget,
        eval_count=len(losses),
        **extra,
    )


class TimeWindowTests(unittest.TestCase):
    def test_late_best_does_not_become_a_timed_result(self):
        report = summarize_time_window(objective([5.0, 2.0, -100.0], [True, True, True], [1.0, 10.0, 10.1]))
        self.assertEqual(report["time_window_best_feasible_loss"], 2.0)
        self.assertEqual(report["time_window_feasible_candidate_count"], 2)
        self.assertEqual(report["after_budget_record_count"], 1)

    def test_exact_boundary_is_included(self):
        report = summarize_time_window(objective([4.0], [True], [10.0]))
        self.assertEqual(report["time_window_best_feasible_loss"], 4.0)

    def test_batch_is_counted_as_candidates_not_one_scalar(self):
        obj = objective(
            [np.array([9.0, 3.0, np.nan]), np.array([-50.0, -60.0])],
            [np.array([False, True, True]), np.array([True, True])],
            [2.0, 11.0],
        )
        report = summarize_time_window(obj)
        self.assertEqual(report["time_window_record_count"], 1)
        self.assertEqual(report["time_window_finite_candidate_count"], 2)
        self.assertEqual(report["time_window_feasible_candidate_count"], 1)
        self.assertEqual(report["time_window_best_feasible_loss"], 3.0)

    def test_short_aux_history_stays_missing(self):
        report = summarize_time_window(objective([8.0, -9.0], [True], [1.0, 2.0]))
        self.assertEqual(report["time_window_best_feasible_loss"], 8.0)
        self.assertEqual(report["time_window_missing_feasibility_candidate_count"], 1)

    def test_no_feasible_result_is_not_zero_loss(self):
        report = summarize_time_window(objective([1.0], [False], [1.0]))
        self.assertIsNone(report["time_window_best_feasible_loss"])
        self.assertEqual(report["time_window_feasible_candidate_count"], 0)

    def test_all_late_is_an_empty_known_window(self):
        report = summarize_time_window(objective([1.0], [True], [11.0]))
        self.assertEqual(report["time_window_status"], "complete")
        self.assertEqual(report["time_window_record_count"], 0)
        self.assertIsNone(report["time_window_best_feasible_loss"])

    def test_empty_history_is_known_empty(self):
        report = summarize_time_window(objective([], [], []))
        self.assertEqual(report["time_window_status"], "complete")
        self.assertEqual(report["time_window_feasible_candidate_count"], 0)

    def test_missing_clock_and_bad_budgets_are_unavailable(self):
        for budget in (None, 0, -1, float("nan"), float("inf"), True, "10"):
            with self.subTest(budget=budget):
                report = summarize_time_window(objective([1.0], [True], [1.0], budget))
                self.assertEqual(report["time_window_status"], "unavailable")
                self.assertIsNone(report["time_window_feasible_candidate_count"])

    def test_bad_or_misaligned_timestamps_are_unavailable(self):
        for times in (
            [],
            [None],
            [-1],
            [float("nan")],
            [float("inf")],
            ["1"],
            [True],
            [np.bool_(True)],
            [np.array([1.0])],
        ):
            with self.subTest(times=times):
                report = summarize_time_window(objective([1.0], [True], times))
                self.assertEqual(report["time_window_status"], "unavailable")
                self.assertIsNone(report["time_window_best_feasible_loss"])

    def test_clock_reversal_is_not_silently_filtered(self):
        report = summarize_time_window(objective([1.0, 2.0], [True, True], [2.0, 1.0]))
        self.assertEqual(report["time_window_reason"], "nonmonotonic_timestamps")

    def test_numpy_scalar_times_are_accepted(self):
        report = summarize_time_window(objective([1.0], [True], [np.float64(2.0)]))
        self.assertEqual(report["time_window_best_feasible_loss"], 1.0)

    def test_nonfinite_loss_is_not_rescued_by_valid_time(self):
        report = summarize_time_window(objective([np.nan, np.inf, 3.0], [True, True, True], [1.0, 2.0, 3.0]))
        self.assertEqual(report["time_window_finite_candidate_count"], 1)
        self.assertEqual(report["time_window_best_feasible_loss"], 3.0)

    def test_state_is_not_mutated(self):
        obj = objective([4.0, -7.0], [True, True], [1.0, 12.0])
        before = deepcopy(vars(obj))
        summarize_time_window(obj)
        self.assertEqual(vars(obj), before)

    def test_nonempty_eval_count_without_history_is_not_zero(self):
        obj = objective([], [], [])
        obj.eval_count = 3
        report = summarize_time_window(obj)
        self.assertEqual(report["time_window_reason"], "missing_loss_history")
