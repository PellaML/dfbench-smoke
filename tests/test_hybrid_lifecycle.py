"""Focused tests for the hybrid optimizer."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from hybrid_fakes import AnalyticObjective, flat, one_evaluation_solve

from dfbench_smoke.algorithms.core import run_hybrid

PROJECT = Path(__file__).resolve().parents[1]


class HybridLifecycleTests(unittest.TestCase):
    def test_matching_aux_warmup_and_logging_happen_exactly_once(self):
        objective = AnalyticObjective(max_evals=10)
        run_hybrid(objective, random_seed=9)
        self.assertEqual(objective.events[:3], ["sample", "warmup_aux", "start_logging"])
        self.assertEqual(objective.events.count("warmup_aux"), 1)
        self.assertEqual(objective.events.count("start_logging"), 1)
        self.assertEqual(objective.events.count("evaluate_aux"), 10)

    def test_zero_budget_has_no_real_evaluations_or_solves(self):
        for budgets in ({"max_evals": 0}, {"max_evals": None, "max_time": 0.0}):
            with self.subTest(budgets=budgets):
                objective = AnalyticObjective(**budgets)
                stats = run_hybrid(objective, np.zeros(2), random_seed=11)
                self.assertEqual(stats["budget_exits"], 1)
                self.assertTrue(all(value == 0 for name, value in stats.items() if name != "budget_exits"))
                self.assertEqual(objective.events, ["warmup_aux", "start_logging"])

    def test_single_evaluation_budget_stops_inside_scipy(self):
        objective = AnalyticObjective(max_evals=1)
        stats = run_hybrid(objective, np.zeros(2), random_seed=11)
        self.assertEqual(stats["real_evaluations"], 1)
        self.assertEqual(stats["local_solves"], 1)
        self.assertEqual(stats["restarts"], 0)

    def test_line_search_budget_exhaustion_stops_cleanly(self):
        objective = AnalyticObjective(max_evals=3)
        stats = run_hybrid(objective, np.array([100.0, -200.0]), random_seed=11)
        self.assertEqual(stats["warm_start_attempts"], 0)
        self.assertEqual(stats["local_solves"], 1)
        self.assertEqual(stats["real_evaluations"], 3)
        self.assertEqual(stats["restarts"], 0)

    def test_time_only_budget_ends_after_an_indivisible_evaluation(self):
        objective = AnalyticObjective(max_evals=None, max_time=0.11, eval_seconds=0.02)
        stats = run_hybrid(objective, np.array([4.0, -3.0]), random_seed=17)
        self.assertEqual(stats["real_evaluations"], 6)
        self.assertEqual(stats["warm_start_attempts"], 1)
        self.assertAlmostEqual(objective.time_elapsed, 0.12)
        self.assertEqual(objective.events.count("start_logging"), 1)

    def test_tightest_time_budget_limits_the_warm_phase(self):
        objective = AnalyticObjective(max_evals=1000, max_time=0.2, eval_seconds=0.02)
        stats = run_hybrid(objective, np.array([100.0, -200.0]), random_seed=17)
        self.assertEqual(stats["warm_start_attempts"], 2)
        self.assertEqual(stats["real_evaluations"], 10)
        self.assertEqual(objective.max_evals, 1000)
        self.assertEqual(objective.max_time, 0.2)

    def test_iteration_callback_checks_budget_without_another_evaluation(self):
        objective = AnalyticObjective(max_evals=6, max_time=2.0)

        def solver(fun, params, *, callback, **kwargs):
            fun(params)
            objective.elapsed_extra = objective.max_time
            callback(params)
            raise AssertionError("Iteration callback should leave the solve")

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, np.zeros(2), random_seed=23)
        self.assertEqual(stats["real_evaluations"], 1)
        self.assertEqual(stats["restarts"], 0)

    def test_eval_fraction_and_twelve_attempt_limit(self):
        def linear(params):
            return float(params.sum()), np.ones_like(params), True

        for cap in (1, 6, 7, 20, 50, 80, 100):
            with (
                self.subTest(cap=cap),
                patch("dfbench_smoke.algorithms.core.minimize", side_effect=one_evaluation_solve),
            ):
                objective = AnalyticObjective(linear, max_evals=cap)
                stats = run_hybrid(objective, np.full(2, 100.0), random_seed=23)
                self.assertEqual(stats["warm_start_attempts"], min(12, cap * 3 // 20))
                self.assertEqual(stats["warm_adam_steps"], stats["warm_start_attempts"])
                self.assertEqual(stats["real_evaluations"], cap)

    def test_clipped_adam_matches_the_fixed_policy(self):
        gradient = np.array([3.0, 4.0])
        objective = AnalyticObjective(lambda params: (float(params @ gradient), gradient, True), max_evals=20)
        initial = np.array([2.0, 3.0])
        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=one_evaluation_solve):
            stats = run_hybrid(objective, initial, random_seed=23)
        clipped = gradient / 5.0
        expected = initial.copy()
        first = np.zeros(2)
        second = np.zeros(2)
        for step in range(1, 4):
            np.testing.assert_allclose(objective.points[step - 1], expected, rtol=0.0, atol=1e-15)
            first = 0.9 * first + 0.1 * clipped
            second = 0.999 * second + 0.001 * clipped**2
            expected -= 0.1 * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
        np.testing.assert_allclose(objective.points[3], expected, rtol=0.0, atol=1e-15)
        self.assertEqual(stats["warm_adam_steps"], 3)
        np.testing.assert_array_equal(initial, [2.0, 3.0])

    def test_constant_sampler_and_flat_function_do_not_stall_the_budget(self):
        objective = AnalyticObjective(flat, max_evals=157)
        objective.sample = np.zeros(2)
        stats = run_hybrid(objective, random_seed=23)
        self.assertEqual(stats["real_evaluations"], 157)
        self.assertEqual(stats["warm_adam_steps"], 1)
        self.assertEqual(stats["local_solves"], 156)
        self.assertEqual(stats["restarts"], 155)
        self.assertGreater(objective.sample_count, 1)
        np.testing.assert_array_equal(objective.sample, np.zeros(2))

    def test_reusing_an_already_logged_objective_does_not_reset_it(self):
        objective = AnalyticObjective(max_evals=6)
        run_hybrid(objective, np.zeros(2), random_seed=61)
        with self.assertRaisesRegex(RuntimeError, "Warmup must run once"):
            run_hybrid(objective, np.zeros(2), random_seed=61)
        self.assertEqual(objective.eval_count, 6)
        self.assertEqual(objective.events.count("start_logging"), 1)

    def test_identical_vectors_at_every_restart_still_consume_real_evaluations(self):
        from scipy.optimize import minimize as scipy_minimize

        objective = AnalyticObjective(flat, max_evals=19)
        objective.sample = np.full(2, np.finfo(float).max)
        invocations = 0

        def guarded_solver(*args, **kwargs):
            nonlocal invocations
            invocations += 1
            self.assertLessEqual(invocations, 19, "Restart cache stalled the evaluation budget")
            return scipy_minimize(*args, **kwargs)

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=guarded_solver):
            stats = run_hybrid(objective, random_seed=67)
        self.assertEqual(stats["real_evaluations"], 19)
        self.assertTrue(all(np.array_equal(point, objective.sample) for point in objective.points))
        self.assertEqual(stats["numerical_failures"], 0)
