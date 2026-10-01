"""Focused tests for the hybrid optimizer."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from hybrid_fakes import AnalyticObjective, flat, one_evaluation_solve

from dfbench_smoke.algorithms.core import run_hybrid

PROJECT = Path(__file__).resolve().parents[1]


class HybridRestartTests(unittest.TestCase):
    def test_repeated_scipy_calls_are_cached_but_each_restart_is_real(self):
        objective = AnalyticObjective(flat, max_evals=8)
        objective.sample = np.zeros(2)

        def repeated_solver(fun, params, **kwargs):
            _, gradient = fun(params)
            gradient[:] = 999.0
            _, repeated_gradient = fun(params.copy())
            np.testing.assert_array_equal(repeated_gradient, np.zeros(2))
            return SimpleNamespace(status=0)

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=repeated_solver):
            stats = run_hybrid(objective, np.zeros(2), random_seed=23)
        self.assertEqual(stats["real_evaluations"], 8)
        self.assertEqual(stats["cache_hits"], 6)
        self.assertEqual(stats["local_solves"], 7)
        self.assertEqual(stats["restarts"], 6)

    def test_restart_schedule_is_two_local_perturbations_then_objective_sample(self):
        objective = AnalyticObjective(flat, max_evals=8)
        objective.sample = np.array([9.0, 9.0])
        initial = np.array([2.0, -3.0])
        stats = run_hybrid(objective, initial, random_seed=31)
        rng = np.random.default_rng(31)
        expected = [initial, initial]
        for _ in range(2):
            expected.extend([initial + rng.normal(0.0, 0.3, 2), initial + rng.normal(0.0, 0.8, 2), objective.sample])
        np.testing.assert_array_equal(np.array(objective.points), np.array(expected))
        self.assertEqual(stats["restarts"], 6)
        self.assertEqual(objective.sample_count, 2)

    def test_unverified_solver_result_does_not_choose_restart_center(self):
        objective = AnalyticObjective(flat, max_evals=6)
        initial = np.array([2.0, -3.0])
        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=one_evaluation_solve):
            run_hybrid(objective, initial, random_seed=31)
        expected = initial + np.random.default_rng(31).normal(0.0, 0.3, 2)
        np.testing.assert_array_equal(objective.points[1], expected)
        self.assertTrue(all(np.all(np.isfinite(point)) for point in objective.points))

    def test_finite_feasible_point_with_bad_gradient_drives_restart(self):
        def constrained(params):
            if np.array_equal(params, [2.0]):
                return 9.0, np.array([np.nan]), True
            return 1.0, np.zeros(1), False

        objective = AnalyticObjective(constrained, size=1, max_evals=6)
        starts = []

        def solver(fun, params, **kwargs):
            starts.append(params.copy())
            fun(params)
            fun(np.array([2.0]))

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, np.array([0.0]), random_seed=37)
        expected = np.array([2.0]) + np.random.default_rng(37).normal(0.0, 0.3, 1)
        np.testing.assert_array_equal(starts[1], expected)
        self.assertEqual(stats["real_evaluations"], 6)
        self.assertEqual(stats["numerical_failures"], 3)

    def test_nonfinite_values_restart_until_budget_without_fake_losses(self):
        for loss, gradient in ((np.inf, np.zeros(2)), (np.nan, np.zeros(2)), (3.0, np.full(2, np.nan))):
            with self.subTest(loss=loss, gradient=gradient):
                objective = AnalyticObjective(
                    lambda _, loss=loss, gradient=gradient: (loss, gradient, True), max_evals=15
                )
                stats = run_hybrid(objective, np.zeros(2), random_seed=37)
                self.assertEqual(stats["real_evaluations"], 15)
                self.assertEqual(stats["numerical_failures"], 15)
                self.assertEqual(stats["warm_start_attempts"], 1)
                self.assertEqual(stats["warm_adam_steps"], 0)
                self.assertEqual(stats["solve_exits_numerical"], 14)
                self.assertEqual(stats["solve_exits_budget"], 0)
                self.assertEqual(stats["budget_exits"], 1)
                np.testing.assert_array_equal(objective.losses, np.full(15, loss))

    def test_nonfinite_solver_trial_aborts_only_that_solve(self):
        objective = AnalyticObjective(flat, max_evals=6)

        def solver(fun, params, **kwargs):
            fun(params)
            fun(np.full(2, np.inf))
            raise AssertionError("Nonfinite trial should abort this solve")

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, np.zeros(2), random_seed=37)
        self.assertEqual(stats["real_evaluations"], 6)
        self.assertEqual(stats["numerical_failures"], 5)
        self.assertEqual(stats["solve_exits_numerical"], 5)
        self.assertEqual(stats["solve_exits_budget"], 1)
        self.assertEqual(stats["budget_exits"], 1)
        self.assertTrue(all(np.all(np.isfinite(point)) for point in objective.points))

    def test_lbfgs_is_unbounded_with_default_iteration_and_evaluation_safeguards(self):
        objective = AnalyticObjective(flat, max_evals=6)
        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=one_evaluation_solve) as solver:
            stats = run_hybrid(objective, np.array([1e100, -1e100]), random_seed=59)
        self.assertEqual(stats["real_evaluations"], 6)
        self.assertGreater(solver.call_count, 1)
        for call in solver.call_args_list:
            self.assertNotIn("bounds", call.kwargs)
            self.assertIs(call.kwargs["jac"], True)
            self.assertEqual(call.kwargs["method"], "L-BFGS-B")
            self.assertNotIn("maxiter", call.kwargs["options"])
            self.assertNotIn("maxfun", call.kwargs["options"])
            self.assertEqual(call.kwargs["options"], {"maxcor": 10, "maxls": 20, "gtol": 1e-6, "ftol": 1e-12})
        self.assertEqual(objective.max_evals, 6)
        self.assertIsNone(objective.max_time)

    def test_exact_constrained_quadratic_restarts_from_a_physically_feasible_point(self):
        def constrained(params):
            return float(params @ params), 2.0 * params, bool(params[0] >= 1.0)

        objective = AnalyticObjective(constrained, size=1, max_evals=6)
        starts = []

        def solver(fun, params, **kwargs):
            starts.append(params.copy())
            fun(params)
            fun(np.zeros(1))
            return SimpleNamespace(status=0)

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, np.array([2.0]), random_seed=71)
        expected = np.array([2.0]) + np.random.default_rng(71).normal(0.0, 0.3, 1)
        np.testing.assert_array_equal(starts[1], expected)
        self.assertEqual(stats["real_evaluations"], 6)
        self.assertEqual(stats["numerical_failures"], 0)
        self.assertIn(True, objective.feasibility)
        self.assertIn(False, objective.feasibility)
