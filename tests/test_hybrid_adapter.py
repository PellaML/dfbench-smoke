"""Adapter contract tests against a fake dfbench module, never the simulator."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ALGORITHMS = PROJECT / "src" / "dfbench_smoke" / "algorithms"


class HybridAdapterTests(unittest.TestCase):
    def setUp(self):
        self.prepare = Mock(return_value=(271, object()))
        self.base = type("OptimizationAlgorithm", (), {"prepare": self.prepare})
        fake = ModuleType("dfbench")
        fake.OptimizationAlgorithm = self.base
        spec = importlib.util.spec_from_file_location("dfbench_smoke.algorithms.hybrid", ALGORITHMS / "hybrid.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"dfbench": fake}):
            spec.loader.exec_module(self.module)
        self.algorithm = self.module.HybridLBFGS()

    def test_class_inherits_framework_algorithm_and_has_its_own_name(self):
        self.assertIsInstance(self.algorithm, self.base)
        self.assertEqual(self.algorithm.algorithm_str, "hybrid_lbfgs")
        self.assertEqual(self.algorithm.diagnostics, {})

    def test_optimize_signature_matches_the_fixed_interface(self):
        signature = inspect.signature(self.module.HybridLBFGS.optimize)
        self.assertEqual(list(signature.parameters), ["self", "objective", "init_params", "random_seed"])
        self.assertIsNone(signature.parameters["init_params"].default)
        self.assertIsNone(signature.parameters["random_seed"].default)

    def test_prepare_runs_once_with_unbounded_space(self):
        objective = object()
        initial = np.array([1.0, 2.0])
        counts = {"real_evaluations": 5}
        with patch.object(self.module, "run_hybrid", return_value=counts) as core:
            result = self.algorithm.optimize(objective, init_params=initial, random_seed=19)
        self.prepare.assert_called_once_with(objective, unbounded=True, random_seed=19)
        core.assert_called_once_with(objective, init_params=initial, random_seed=271)
        self.assertIs(core.call_args.kwargs["init_params"], initial)
        self.assertIsNone(result)
        self.assertEqual(self.algorithm.diagnostics, counts)
        self.assertEqual(json.loads(json.dumps(self.algorithm.diagnostics, allow_nan=False)), counts)

    def test_default_seed_is_resolved_only_by_prepare(self):
        objective = object()
        with patch.object(self.module, "run_hybrid", return_value={}) as core:
            self.algorithm.optimize(objective)
        self.prepare.assert_called_once_with(objective, unbounded=True, random_seed=None)
        core.assert_called_once_with(objective, init_params=None, random_seed=271)

    def test_zero_seed_is_not_treated_as_missing(self):
        objective = object()
        self.prepare.return_value = (0, object())
        with patch.object(self.module, "run_hybrid", return_value={}) as core:
            self.algorithm.optimize(objective, random_seed=0)
        self.prepare.assert_called_once_with(objective, unbounded=True, random_seed=0)
        core.assert_called_once_with(objective, init_params=None, random_seed=0)

    def test_diagnostics_are_replaced_for_each_independent_run(self):
        with patch.object(self.module, "run_hybrid", side_effect=[{"real_evaluations": 5}, {"real_evaluations": 9}]):
            self.algorithm.optimize(object(), random_seed=3)
            self.assertEqual(self.algorithm.diagnostics, {"real_evaluations": 5})
            self.algorithm.optimize(object(), random_seed=7)
            self.assertEqual(self.algorithm.diagnostics, {"real_evaluations": 9})
        self.assertEqual(self.prepare.call_count, 2)

    def test_core_error_propagates_without_stale_diagnostics(self):
        self.algorithm.diagnostics = {"real_evaluations": 99}
        with (
            patch.object(self.module, "run_hybrid", side_effect=ValueError("bad input")),
            self.assertRaisesRegex(ValueError, "bad input"),
        ):
            self.algorithm.optimize(object(), random_seed=3)
        self.assertEqual(self.algorithm.diagnostics, {})
        self.assertEqual(self.prepare.call_count, 1)

    def test_prepare_error_prevents_core_execution(self):
        self.prepare.side_effect = RuntimeError("invalid Objective state")
        with (
            patch.object(self.module, "run_hybrid") as core,
            self.assertRaisesRegex(RuntimeError, "invalid Objective state"),
        ):
            self.algorithm.optimize(object(), random_seed=3)
        core.assert_not_called()
        self.assertEqual(self.algorithm.diagnostics, {})

    def test_package_marker_has_no_eager_imports(self):
        tree = ast.parse((ALGORITHMS / "__init__.py").read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports, [])


if __name__ == "__main__":
    unittest.main()
