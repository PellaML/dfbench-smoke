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


class HybridValidationTests(unittest.TestCase):
    def test_invalid_initial_vectors_fail_before_warmup(self):
        for initial in ([], [1.0], [[1.0, 2.0]], [np.nan, 1.0], [np.inf, 1.0], [1j, 2j], ["1", "2"], [True, False]):
            with self.subTest(initial=initial):
                objective = AnalyticObjective()
                with self.assertRaises((ValueError, TypeError)):
                    run_hybrid(objective, initial, random_seed=43)
                self.assertEqual(objective.events, [])

    def test_invalid_objective_sample_fails_before_warmup(self):
        objective = AnalyticObjective()
        objective.sample = np.array([np.inf, 0.0])
        with self.assertRaisesRegex(ValueError, "initial parameters"):
            run_hybrid(objective, random_seed=43)
        self.assertEqual(objective.events, ["sample"])

    def test_invalid_restart_sample_is_not_silently_replaced(self):
        objective = AnalyticObjective(flat, max_evals=6)
        objective.sample = np.array([[0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "restart parameters"):
            run_hybrid(objective, np.zeros(2), random_seed=43)
        self.assertEqual(objective.eval_count, 3)
        self.assertEqual(objective.events.count("start_logging"), 1)

    def test_readonly_initial_vector_and_shared_gradient_are_not_mutated(self):
        initial = np.array([3.0, 4.0])
        initial.flags.writeable = False
        gradient = np.array([1.0, -1.0])
        gradient.flags.writeable = False
        objective = AnalyticObjective(lambda params: (float(params @ gradient), gradient, True), max_evals=20)
        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=one_evaluation_solve):
            run_hybrid(objective, initial, random_seed=43)
        np.testing.assert_array_equal(initial, [3.0, 4.0])
        np.testing.assert_array_equal(gradient, [1.0, -1.0])

    def test_solver_mutations_do_not_rewrite_objective_parameter_references(self):
        objective = AnalyticObjective(flat, max_evals=20)
        initial = np.array([2.0, -3.0])

        def mutating_solver(fun, params, **kwargs):
            fun(params)
            self.assertFalse(np.shares_memory(params, objective.retained_points[-1]))
            params[:] += 100.0
            fun(params)
            params[:] = -999.0
            return SimpleNamespace(status=2)

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=mutating_solver):
            run_hybrid(objective, initial, random_seed=73)
        np.testing.assert_array_equal(initial, [2.0, -3.0])
        np.testing.assert_array_equal(objective.retained_points, objective.points)
        self.assertTrue(
            all(
                not np.shares_memory(left, right)
                for left, right in zip(objective.retained_points, objective.points, strict=True)
            )
        )
        self.assertEqual(objective.eval_count, 20)

    def test_invalid_public_configuration_is_rejected_before_logging(self):
        for name, value in (
            ("n_params", 0),
            ("n_params", 1.5),
            ("n_params", True),
            ("max_evals", -1),
            ("max_evals", 2.5),
            ("max_evals", True),
            ("max_time", -1.0),
            ("max_time", np.inf),
            ("max_time", np.nan),
            ("max_time", True),
        ):
            with self.subTest(name=name, value=value):
                objective = AnalyticObjective()
                setattr(objective, name, value)
                with self.assertRaisesRegex(ValueError, name):
                    run_hybrid(objective, np.zeros(2), random_seed=59)
                self.assertEqual(objective.events, [])

    def test_missing_finite_budget_is_explicit_not_an_unbounded_loop(self):
        objective = AnalyticObjective(max_evals=None, max_time=None)
        with self.assertRaisesRegex(ValueError, "finite Objective"):
            run_hybrid(objective, np.zeros(2), random_seed=59)
        self.assertEqual(objective.events, [])

    def test_core_requires_the_callers_resolved_integer_seed(self):
        for seed in (None, -1, 1.5, True):
            with self.subTest(seed=seed):
                objective = AnalyticObjective()
                with self.assertRaisesRegex(ValueError, "random_seed"):
                    run_hybrid(objective, np.zeros(2), random_seed=seed)
                self.assertEqual(objective.events, [])

    def test_objective_and_scipy_api_errors_are_not_swallowed(self):
        def broken(_):
            raise RuntimeError("objective API error")

        objective = AnalyticObjective(broken, max_evals=6)
        with self.assertRaisesRegex(RuntimeError, "objective API error"):
            run_hybrid(objective, np.zeros(2), random_seed=61)
        with (
            patch("dfbench_smoke.algorithms.core.minimize", side_effect=ValueError("solver programming error")),
            self.assertRaisesRegex(ValueError, "solver programming error"),
        ):
            run_hybrid(AnalyticObjective(max_evals=6), np.zeros(2), random_seed=61)
