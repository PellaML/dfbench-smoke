"""Organizer feasible-best summary, exercised with fake public Objective state.

The vendored function is tested as shipped. These fakes expose only the public
properties it reads, so no simulator, JAX or dfbench import is needed.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dfbench_smoke.vendor import validate_submission

SRC = Path(__file__).resolve().parents[1] / "src"
summarize = validate_submission.summarize_objective
NAN = float("nan")
INF = float("inf")


def fake(losses, feasibility, time_elapsed=12.5, eval_count=None, budget_exceeded=False):
    return SimpleNamespace(
        loss_history=list(losses),
        is_feasible_history=list(feasibility),
        time_elapsed=time_elapsed,
        eval_count=len(losses) if eval_count is None else eval_count,
        budget_exceeded=budget_exceeded,
    )


class ImportBoundaryTest(unittest.TestCase):
    def test_summary_import_does_not_load_simulator_stack(self):
        # A fresh interpreter keeps the check independent of other test modules.
        probe = (
            "import sys; from dfbench_smoke.vendor import validate_submission; "
            "print(','.join(n for n in ('jax', 'jaxlib', 'dfbench', 'differometor') if n in sys.modules))"
        )
        env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=60, check=True
        )
        self.assertEqual(completed.stdout.strip(), "")


class ScalarHistoryTest(unittest.TestCase):
    def test_best_is_minimum_over_feasible_finite_losses_only(self):
        result = summarize(fake([3.0, -1.0, 0.5, -7.0], [True, True, True, False]))
        self.assertEqual(result["best_feasible_loss"], -1.0)
        self.assertEqual(result["finite_candidate_count"], 4)
        self.assertEqual(result["feasible_candidate_count"], 3)
        self.assertEqual(result["missing_feasibility_candidate_count"], 0)
        self.assertIsInstance(result["best_feasible_loss"], float)

    def test_no_feasible_entry_gives_none_not_an_infeasible_loss(self):
        result = summarize(fake([-5.0, -9.0], [False, False]))
        self.assertIsNone(result["best_feasible_loss"])
        self.assertEqual(result["feasible_candidate_count"], 0)
        self.assertEqual(result["finite_candidate_count"], 2)

    def test_nan_and_infinite_losses_are_never_candidates(self):
        result = summarize(fake([NAN, INF, -INF, 2.0], [True, True, True, True]))
        self.assertEqual(result["best_feasible_loss"], 2.0)
        self.assertEqual(result["finite_candidate_count"], 1)
        self.assertEqual(result["feasible_candidate_count"], 1)

    def test_only_nonfinite_feasible_losses_give_none(self):
        result = summarize(fake([NAN, -INF], [True, True]))
        self.assertIsNone(result["best_feasible_loss"])
        self.assertEqual(result["finite_candidate_count"], 0)
        self.assertEqual(result["missing_feasibility_candidate_count"], 0)

    def test_numpy_scalar_types_are_accepted(self):
        result = summarize(fake([np.float64(1.5), np.float32(0.25)], [np.bool_(True), np.True_]))
        self.assertEqual(result["best_feasible_loss"], 0.25)
        self.assertEqual(result["feasible_candidate_count"], 2)


class MissingAuxTest(unittest.TestCase):
    def test_none_feasibility_is_counted_missing_not_infeasible(self):
        result = summarize(fake([-4.0, 1.0], [None, True]))
        self.assertEqual(result["best_feasible_loss"], 1.0)
        self.assertEqual(result["missing_feasibility_candidate_count"], 1)
        self.assertEqual(result["feasible_candidate_count"], 1)

    def test_feasibility_history_shorter_than_losses(self):
        result = summarize(fake([2.0, -3.0, -8.0], [True]))
        self.assertEqual(result["best_feasible_loss"], 2.0)
        self.assertEqual(result["missing_feasibility_candidate_count"], 2)

    def test_missing_feasibility_on_nonfinite_loss_is_not_counted(self):
        result = summarize(fake([NAN, 1.0], [None, True]))
        self.assertEqual(result["missing_feasibility_candidate_count"], 0)
        self.assertEqual(result["finite_candidate_count"], 1)

    def test_extra_feasibility_entries_are_ignored(self):
        result = summarize(fake([1.0], [True, True, True]))
        self.assertEqual(result["feasible_candidate_count"], 1)
        self.assertEqual(result["missing_feasibility_candidate_count"], 0)

    def test_batch_size_mismatch_is_missing_and_never_feasible(self):
        result = summarize(fake([np.array([-2.0, -6.0, NAN])], [np.array([True, True])]))
        self.assertIsNone(result["best_feasible_loss"])
        self.assertEqual(result["finite_candidate_count"], 2)
        self.assertEqual(result["missing_feasibility_candidate_count"], 2)
        self.assertEqual(result["feasible_candidate_count"], 0)

    def test_scalar_flag_against_batch_is_a_mismatch(self):
        result = summarize(fake([np.array([-1.0, -2.0])], [True]))
        self.assertIsNone(result["best_feasible_loss"])
        self.assertEqual(result["missing_feasibility_candidate_count"], 2)

    def test_no_feasibility_history_at_all(self):
        result = summarize(fake([-1.0, -2.0], []))
        self.assertIsNone(result["best_feasible_loss"])
        self.assertEqual(result["missing_feasibility_candidate_count"], 2)
        self.assertEqual(result["feasible_candidate_count"], 0)


class BatchedHistoryTest(unittest.TestCase):
    def test_batched_entries_use_elementwise_feasibility(self):
        losses = [np.array([5.0, -10.0, -2.0]), np.array([-3.0, NAN])]
        flags = [np.array([True, False, True]), [True, True]]
        result = summarize(fake(losses, flags))
        self.assertEqual(result["best_feasible_loss"], -3.0)
        self.assertEqual(result["finite_candidate_count"], 4)
        self.assertEqual(result["feasible_candidate_count"], 3)

    def test_mixed_scalar_and_batched_history(self):
        losses = [0.75, np.array([[1.0, -0.5]]), np.float64(-0.25)]
        flags = [True, np.array([[True, True]]), False]
        result = summarize(fake(losses, flags))
        self.assertEqual(result["best_feasible_loss"], -0.5)
        self.assertEqual(result["feasible_candidate_count"], 3)
        self.assertEqual(result["finite_candidate_count"], 4)

    def test_infeasible_batch_minimum_is_not_reported(self):
        result = summarize(fake([np.array([-100.0, 4.0])], [np.array([False, True])]))
        self.assertEqual(result["best_feasible_loss"], 4.0)

    def test_batch_of_only_nonfinite_values(self):
        result = summarize(fake([np.array([NAN, INF])], [np.array([True, True])]))
        self.assertIsNone(result["best_feasible_loss"])
        self.assertEqual(result["finite_candidate_count"], 0)


class ObjectiveFieldsTest(unittest.TestCase):
    def test_empty_histories(self):
        result = summarize(fake([], [], time_elapsed=0.0, eval_count=0))
        self.assertEqual(
            result,
            {
                "objective_time_seconds": 0.0,
                "evaluation_count": 0,
                "budget_exceeded": False,
                "finite_candidate_count": 0,
                "feasible_candidate_count": 0,
                "missing_feasibility_candidate_count": 0,
                "best_feasible_loss": None,
            },
        )

    def test_object_without_public_properties_uses_defaults(self):
        result = summarize(object())
        self.assertIsNone(result["objective_time_seconds"])
        self.assertEqual(result["evaluation_count"], 0)
        self.assertFalse(result["budget_exceeded"])
        self.assertIsNone(result["best_feasible_loss"])

    def test_time_eval_and_budget_are_copied_and_coerced(self):
        result = summarize(
            fake([1.0], [True], time_elapsed=np.float64(120.25), eval_count=np.int64(7), budget_exceeded=np.True_)
        )
        self.assertEqual(result["objective_time_seconds"], 120.25)
        self.assertEqual(result["evaluation_count"], 7)
        self.assertIs(type(result["evaluation_count"]), int)
        self.assertIs(result["budget_exceeded"], True)

    def test_eval_count_is_reported_not_derived_from_history(self):
        result = summarize(fake([1.0], [True], eval_count=40))
        self.assertEqual(result["evaluation_count"], 40)

    def test_nonfinite_or_invalid_time_becomes_none(self):
        for value in (NAN, INF, None, "soon"):
            with self.subTest(value=value):
                self.assertIsNone(summarize(fake([], [], time_elapsed=value))["objective_time_seconds"])

    def test_histories_are_read_not_mutated(self):
        losses, flags = [1.0, np.array([2.0, NAN])], [True, np.array([True, False])]
        objective = fake(losses, flags)
        summarize(objective)
        self.assertEqual(objective.loss_history[0], 1.0)
        self.assertTrue(math.isnan(objective.loss_history[1][1]))
        self.assertEqual(len(objective.is_feasible_history), 2)


if __name__ == "__main__":
    unittest.main()
