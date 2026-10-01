"""Focused tests for the hybrid optimizer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
import warnings
from pathlib import Path

import numpy as np
from hybrid_fakes import AnalyticObjective, flat

from dfbench_smoke.algorithms.core import _clip_gradient, run_hybrid

PROJECT = Path(__file__).resolve().parents[1]


class HybridBehaviorTests(unittest.TestCase):
    def test_quadratic_reaches_its_minimum_and_uses_the_full_budget(self):
        objective = AnalyticObjective()
        stats = run_hybrid(objective, np.array([4.0, -3.0]), random_seed=17)
        self.assertLess(min(objective.losses), 1e-12)
        self.assertEqual(objective.eval_count, objective.max_evals)
        self.assertEqual(stats["real_evaluations"], objective.eval_count)
        self.assertGreater(stats["local_solves"], 0)
        self.assertGreater(stats["restarts"], 0)
        self.assertEqual(stats["numerical_failures"], 0)

    def test_ill_conditioned_analytic_quadratic(self):
        def ill_conditioned(params):
            residual = params - np.array([1.0, -2.0, 0.5])
            weights = np.array([1.0, 30.0, 300.0])
            return float((weights * residual) @ residual), 2.0 * weights * residual, True

        objective = AnalyticObjective(ill_conditioned, size=3, max_evals=120)
        stats = run_hybrid(objective, np.array([7.0, 5.0, -1.0]), random_seed=29)
        self.assertLess(min(objective.losses), 1e-8)
        self.assertEqual(stats["real_evaluations"], 120)
        self.assertEqual(stats["numerical_failures"], 0)

    def test_multimodal_analytic_objective_leaves_a_stationary_start(self):
        def double_well(params):
            x, y = params
            return (x * x - 1.0) ** 2 + 0.5 * (y - 2.0) ** 2, np.array([4 * x * (x * x - 1), y - 2]), True

        objective = AnalyticObjective(double_well, max_evals=70)
        stats = run_hybrid(objective, np.array([0.0, 2.0]), random_seed=137)
        self.assertLess(min(objective.losses), 1e-8)
        self.assertGreater(stats["restarts"], 0)
        self.assertEqual(stats["real_evaluations"], 70)

    def test_identical_seeds_have_identical_trajectories(self):
        objectives = [AnalyticObjective(flat, max_evals=20, seed=47) for _ in range(2)]
        stats = [run_hybrid(objective, random_seed=47) for objective in objectives]
        np.testing.assert_array_equal(objectives[0].points, objectives[1].points)
        self.assertEqual(stats[0], stats[1])
        different = AnalyticObjective(flat, max_evals=20, seed=47)
        run_hybrid(different, random_seed=53)
        self.assertFalse(np.array_equal(different.points, objectives[0].points))

    def test_core_does_not_reseed_numpy_global_rng(self):
        before = np.random.get_state()
        run_hybrid(AnalyticObjective(flat, max_evals=6), np.zeros(2), random_seed=59)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_diagnostics_are_small_json_safe_counters_only(self):
        stats = run_hybrid(AnalyticObjective(max_evals=10), np.zeros(2), random_seed=61)
        self.assertEqual(
            set(stats),
            {
                "real_evaluations",
                "cache_hits",
                "warm_start_attempts",
                "warm_adam_steps",
                "local_solves",
                "restarts",
                "numerical_failures",
                "solve_exits_converged",
                "solve_exits_limited",
                "solve_exits_abnormal",
                "solve_exits_numerical",
                "solve_exits_budget",
                "budget_exits",
            },
        )
        self.assertTrue(all(type(value) is int and value >= 0 for value in stats.values()))
        self.assertEqual(json.loads(json.dumps(stats, allow_nan=False)), stats)

    def test_core_and_package_import_without_the_simulator_stack(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            forbidden = {'dfbench', 'jax', 'jaxlib', 'optax', 'differometor', 'learn2design'}
            class BlockSimulatorImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in forbidden:
                        raise AssertionError('Forbidden simulator import: ' + fullname)
                    return None
            sys.meta_path.insert(0, BlockSimulatorImports())
            import dfbench_smoke.algorithms
            assert 'dfbench_smoke.algorithms.hybrid' not in sys.modules
            import dfbench_smoke.algorithms.core
            assert not forbidden.intersection(name.split('.')[0] for name in sys.modules)
            print('core imports without simulator stack')
            """
        )
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(PROJECT / "src"))
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("core imports without simulator stack", result.stdout)


class GradientClippingTests(unittest.TestCase):
    def test_global_l2_clipping(self):
        gradient = np.array([3.0, 4.0])
        np.testing.assert_allclose(_clip_gradient(gradient), [0.6, 0.8])
        np.testing.assert_array_equal(gradient, [3.0, 4.0])

    def test_small_and_zero_gradients_are_not_amplified(self):
        for gradient in (np.zeros(2), np.array([0.1, -0.2]), np.array([np.nextafter(0.0, 1.0), 0.0])):
            with self.subTest(gradient=gradient):
                clipped = _clip_gradient(gradient)
                np.testing.assert_array_equal(clipped, gradient)
                self.assertFalse(np.shares_memory(clipped, gradient))

    def test_extreme_gradient_does_not_overflow(self):
        maximum = np.finfo(np.float64).max
        gradient = np.array([maximum, -maximum, 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            clipped = _clip_gradient(gradient)
        self.assertTrue(np.all(np.isfinite(clipped)))
        np.testing.assert_allclose(clipped, [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(clipped)), 1.0)
        np.testing.assert_array_equal(gradient, [maximum, -maximum, 0.0])
