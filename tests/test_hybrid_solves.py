"""Local-solve exit handling and continuation tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from hybrid_fakes import AnalyticObjective, flat

from dfbench_smoke.algorithms.core import run_hybrid

PROJECT = Path(__file__).resolve().parents[1]


class SolveExitTests(unittest.TestCase):
    def assert_exit_partition(self, stats):
        reasons = ("converged", "limited", "abnormal", "numerical", "budget")
        self.assertEqual(stats["local_solves"], sum(stats[f"solve_exits_{reason}"] for reason in reasons))
        self.assertEqual(stats["budget_exits"], 1)

    def test_limit_continues_from_observed_best_without_reading_result_x(self):
        objective = AnalyticObjective(lambda params: (float(params @ params), 2.0 * params, True), size=1, max_evals=6)
        initial = np.array([3.0])
        starts = []

        class LimitedResult:
            status = 1

            @property
            def x(self):
                raise AssertionError("An unevaluated solver result must not choose the next start")

        def solver(fun, params, **kwargs):
            starts.append(params.copy())
            self.assertLessEqual(len(starts), 3, "Limit continuation must advance the budget")
            fun(params)
            fun(np.array([1.0]) if len(starts) == 1 else np.array([2.0]))
            params[:] = -999.0
            return LimitedResult()

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, initial, random_seed=79)
        np.testing.assert_array_equal(starts, [[3.0], [1.0], [1.0]])
        np.testing.assert_array_equal(initial, [3.0])
        np.testing.assert_array_equal(objective.retained_points, objective.points)
        self.assertEqual(stats["real_evaluations"], 6)
        self.assertEqual(stats["solve_exits_limited"], 2)
        self.assertEqual(stats["solve_exits_budget"], 1)
        self.assertEqual(stats["restarts"], 0)
        self.assertEqual(objective.sample_count, 0)
        self.assert_exit_partition(stats)

    def test_limit_continuation_preserves_feasible_preference(self):
        def constrained(params):
            return float(params @ params), 2.0 * params, bool(params[0] >= 1.0)

        objective = AnalyticObjective(constrained, size=1, max_evals=6)
        starts = []

        def solver(fun, params, **kwargs):
            starts.append(params.copy())
            fun(params)
            fun(np.zeros(1))
            return SimpleNamespace(status=1, x=np.array([999.0]), fun=-1e99)

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, np.array([2.0]), random_seed=79)
        np.testing.assert_array_equal(starts, [[2.0], [2.0], [2.0]])
        self.assertEqual(stats["solve_exits_limited"], 2)
        self.assertEqual(stats["restarts"], 0)
        self.assert_exit_partition(stats)

    def test_repeated_limits_at_identical_points_cannot_spin_on_the_cache(self):
        objective = AnalyticObjective(flat, max_evals=6)
        starts = []
        initial = np.array([2.0, -3.0])

        def solver(fun, params, **kwargs):
            starts.append(params.copy())
            self.assertLessEqual(len(starts), 6, "Repeated limits stalled the evaluation budget")
            fun(params)
            fun(params.copy())
            return SimpleNamespace(status=1)

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, initial, random_seed=79)
        np.testing.assert_array_equal(starts, np.tile(initial, (6, 1)))
        self.assertEqual(stats["real_evaluations"], 6)
        self.assertEqual(stats["cache_hits"], 5)
        self.assertEqual(stats["solve_exits_limited"], 5)
        self.assertEqual(stats["solve_exits_budget"], 1)
        self.assertEqual(stats["restarts"], 0)
        self.assertEqual(objective.sample_count, 0)
        self.assert_exit_partition(stats)

    def test_limits_do_not_consume_restart_rng_or_advance_the_local_local_global_cycle(self):
        objective = AnalyticObjective(flat, max_evals=6)
        objective.sample = np.array([5.0, 6.0])
        initial = np.array([2.0, -3.0])
        statuses = iter((1, 1, 0, 2, 2))

        def solver(fun, params, **kwargs):
            fun(params)
            return SimpleNamespace(status=next(statuses))

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, initial, random_seed=79)
        rng = np.random.default_rng(79)
        expected = [
            initial,
            initial,
            initial,
            initial + rng.normal(0.0, 0.3, 2),
            initial + rng.normal(0.0, 0.8, 2),
            objective.sample,
        ]
        np.testing.assert_array_equal(objective.points, expected)
        self.assertEqual(stats["solve_exits_limited"], 2)
        self.assertEqual(stats["solve_exits_converged"], 1)
        self.assertEqual(stats["solve_exits_abnormal"], 2)
        self.assertEqual(stats["solve_exits_budget"], 1)
        self.assertEqual(stats["restarts"], 3)
        self.assertEqual(objective.sample_count, 1)
        self.assert_exit_partition(stats)

    def test_limit_return_after_deadline_does_not_start_a_continuation(self):
        objective = AnalyticObjective(flat, max_evals=6, max_time=1.0)

        def solver(fun, params, **kwargs):
            fun(params)
            objective.elapsed_extra = 1.0
            return SimpleNamespace(status=1)

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver) as minimize_mock:
            stats = run_hybrid(objective, np.zeros(2), random_seed=79)
        minimize_mock.assert_called_once()
        self.assertEqual(stats["real_evaluations"], 1)
        self.assertEqual(stats["solve_exits_limited"], 1)
        self.assertEqual(stats["solve_exits_budget"], 0)
        self.assertEqual(stats["restarts"], 0)
        self.assert_exit_partition(stats)

    def test_convergence_and_abnormal_exits_keep_perturbed_restarts(self):
        for status, reason in ((0, "converged"), (2, "abnormal")):
            with self.subTest(status=status):
                objective = AnalyticObjective(flat, max_evals=6)

                def solver(fun, params, status=status, **kwargs):
                    fun(params)
                    return SimpleNamespace(status=status)

                with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
                    stats = run_hybrid(objective, np.zeros(2), random_seed=79)
                expected = np.random.default_rng(79).normal(0.0, 0.3, 2)
                np.testing.assert_array_equal(objective.points[1], expected)
                self.assertEqual(stats[f"solve_exits_{reason}"], 5)
                self.assertEqual(stats["solve_exits_budget"], 1)
                self.assertEqual(stats["restarts"], 5)
                self.assert_exit_partition(stats)

    def test_overall_budget_exit_during_warmup_is_not_a_solve_exit(self):
        objective = AnalyticObjective(max_evals=None, max_time=0.005, eval_seconds=0.01)
        stats = run_hybrid(objective, np.zeros(2), random_seed=79)
        self.assertEqual(stats["real_evaluations"], 1)
        self.assertEqual(stats["warm_start_attempts"], 1)
        self.assertEqual(stats["warm_adam_steps"], 0)
        self.assertEqual(stats["local_solves"], 0)
        self.assert_exit_partition(stats)

    def test_budget_interruption_inside_scipy_has_its_own_exit_counter(self):
        stats = run_hybrid(AnalyticObjective(max_evals=1), np.zeros(2), random_seed=79)
        self.assertEqual(stats["real_evaluations"], 1)
        self.assertEqual(stats["local_solves"], 1)
        self.assertEqual(stats["solve_exits_budget"], 1)
        self.assert_exit_partition(stats)

    def test_no_evaluation_return_fails_explicitly_instead_of_spinning(self):
        for status in (0, 1, 2):
            with self.subTest(status=status):
                objective = AnalyticObjective(flat, max_evals=6)
                with (
                    patch(
                        "dfbench_smoke.algorithms.core.minimize", return_value=SimpleNamespace(status=status)
                    ) as solver,
                    self.assertRaisesRegex(RuntimeError, "without an Objective evaluation"),
                ):
                    run_hybrid(objective, np.zeros(2), random_seed=79)
                solver.assert_called_once()
                self.assertEqual(objective.eval_count, 0)
                self.assertEqual(objective.sample_count, 0)
                self.assertEqual(objective.events.count("start_logging"), 1)

    def test_invalid_solver_status_is_an_api_error_not_a_numerical_restart(self):
        for status in (3, -1, 0.5, "1", None, True):
            with self.subTest(status=status):
                objective = AnalyticObjective(flat, max_evals=6)

                def solver(fun, params, status=status, **kwargs):
                    fun(params)
                    return SimpleNamespace(status=status)

                with (
                    patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver) as minimize_mock,
                    self.assertRaisesRegex(ValueError, "L-BFGS-B status"),
                ):
                    run_hybrid(objective, np.zeros(2), random_seed=79)
                minimize_mock.assert_called_once()
                self.assertEqual(objective.eval_count, 1)
                self.assertEqual(objective.sample_count, 0)

    def test_real_scipy_limit_status_is_handled_by_continuation(self):
        from scipy.optimize import minimize as scipy_minimize

        objective = AnalyticObjective(max_evals=20)
        returned_statuses = []

        def one_iteration_solver(fun, params, **kwargs):
            self.assertNotIn("maxiter", kwargs["options"])
            # This limit is a branch fixture, not a production policy or benchmark.
            kwargs["options"] = dict(kwargs["options"], maxiter=1)
            result = scipy_minimize(fun, params, **kwargs)
            returned_statuses.append(result.status)
            return result

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=one_iteration_solver):
            stats = run_hybrid(objective, np.array([4.0, -3.0]), random_seed=79)
        self.assertIn(1, returned_statuses)
        self.assertEqual(stats["solve_exits_limited"], returned_statuses.count(1))
        self.assertEqual(stats["real_evaluations"], 20)
        self.assert_exit_partition(stats)

    def test_analytic_rosenbrock_can_continue_beyond_sixty_four_iterations(self):
        from scipy.optimize import minimize as scipy_minimize
        from scipy.optimize import rosen, rosen_der

        objective = AnalyticObjective(
            lambda params: (float(rosen(params)), rosen_der(params), True), size=30, max_evals=300
        )
        iterations = []

        def solver(fun, params, *, callback, **kwargs):
            index = len(iterations)
            iterations.append(0)

            def counted_callback(point):
                iterations[index] += 1
                callback(point)

            return scipy_minimize(fun, params, callback=counted_callback, **kwargs)

        with patch("dfbench_smoke.algorithms.core.minimize", side_effect=solver):
            stats = run_hybrid(objective, np.full(30, -1.2), random_seed=83)
        self.assertGreater(iterations[0], 64)
        self.assertEqual(stats["solve_exits_limited"], 0)
        self.assertEqual(stats["real_evaluations"], 300)
        self.assertEqual(stats["numerical_failures"], 0)
        self.assertLess(min(objective.losses), objective.losses[0])
        self.assert_exit_partition(stats)
