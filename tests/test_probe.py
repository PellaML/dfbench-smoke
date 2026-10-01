import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from dfbench_smoke.config import ProbeConfig
from dfbench_smoke.probe import run_probe, run_reference


class ProbeTests(unittest.TestCase):
    def test_reference_is_called_once_with_only_the_documented_seed(self):
        objective = SimpleNamespace(
            loss_history=[4.0, 1.0],
            is_feasible_history=[True, False],
            batched_loss_history=[],
            batched_is_feasible_history=[],
            time_elapsed=1.0,
            eval_count=2,
            budget_exceeded=True,
        )
        optimizer = Mock(algorithm_str="reference")
        result = run_reference(objective, optimizer, 42)
        optimizer.optimize.assert_called_once_with(objective, random_seed=42)
        self.assertEqual(result["evaluation_count"], 2)
        self.assertEqual(result["feasible_candidate_count"], 1)
        self.assertEqual(result["best_feasible_loss"], 4.0)
        self.assertEqual(result["algorithm"], "reference")
        self.assertGreaterEqual(result["optimizer_wall_seconds"], 0)

    def test_exception_does_not_turn_into_a_success_report(self):
        optimizer = Mock()
        optimizer.optimize.side_effect = RuntimeError("simulation failed")
        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
            run_reference(object(), optimizer, 42)

    def test_linux_setup_matches_the_organizer_public_runner(self):
        for problem_name, method in (("cvoyager", "adam"), ("uifo", "random")):
            with self.subTest(problem=problem_name, method=method):
                objective = SimpleNamespace(
                    n_params=7,
                    loss_history=[1.0],
                    is_feasible_history=[True],
                    time_elapsed=0.1,
                    eval_count=1,
                    budget_exceeded=False,
                )
                optimizer = Mock(algorithm_str=method)
                jax = ModuleType("jax")
                jax.default_backend = lambda: "cpu"
                jax.config = SimpleNamespace(jax_enable_x64=True)
                framework = ModuleType("dfbench")
                framework.Objective = Mock(return_value=objective)
                problems = ModuleType("dfbench.problems")
                problems.ConstrainedVoyagerProblem = Mock(return_value=object())
                problems.UIFOProblem = Mock(return_value=object())
                resource = ModuleType("resource")
                resource.RUSAGE_SELF = 0
                resource.getrusage = lambda _: SimpleNamespace(ru_maxrss=123)
                adam = ModuleType("dfbench_smoke.vendor.adam_gd")
                adam.AdamGD = Mock(return_value=optimizer)
                random = ModuleType("dfbench_smoke.vendor.random_search")
                random.RandomSearch = Mock(return_value=optimizer)
                modules = {
                    "jax": jax,
                    "dfbench": framework,
                    "dfbench.problems": problems,
                    "resource": resource,
                    "dfbench_smoke.vendor.adam_gd": adam,
                    "dfbench_smoke.vendor.random_search": random,
                }
                with (
                    patch.dict(sys.modules, modules),
                    patch("dfbench_smoke.probe.platform.system", return_value="Linux"),
                    patch("dfbench_smoke.probe.version", return_value="test-version"),
                ):
                    result = run_probe(ProbeConfig(problem_name, method, 120, 41))
                settings = framework.Objective.call_args.kwargs
                self.assertEqual(settings["save"], ["is_feasible", "batched_loss", "batched_is_feasible"])
                self.assertFalse(settings["save_params_history"])
                self.assertFalse(settings["save_batched_params_history"])
                self.assertEqual(settings["max_time"], 120)
                self.assertEqual(settings["display_mode"], "log")
                optimizer.optimize.assert_called_once_with(objective, random_seed=41)
                if problem_name == "uifo":
                    problems.UIFOProblem.assert_called_once_with(topology_seed=41)
                    problems.ConstrainedVoyagerProblem.assert_not_called()
                    self.assertEqual(result["topology_seed"], 41)
                else:
                    problems.ConstrainedVoyagerProblem.assert_called_once_with()
                    problems.UIFOProblem.assert_not_called()
                    self.assertIsNone(result["topology_seed"])
                self.assertEqual(result["peak_process_rss_bytes"], 123 * 1024)
                self.assertEqual(result["parameter_count"], 7)
                self.assertGreaterEqual(result["probe_wall_seconds"], result["optimizer_wall_seconds"])
