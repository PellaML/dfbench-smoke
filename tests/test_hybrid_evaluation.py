"""Cheap boundary tests with no dfbench or simulator imports."""

from __future__ import annotations

import unittest

import numpy as np

from dfbench_smoke.algorithms._evaluation import (
    BudgetExhausted,
    Diagnostics,
    EvaluationSession,
    NumericalFailure,
)


class StubObjective:
    def __init__(self):
        self.result = (3.0, np.array([1.0, 2.0]), {"is_feasible": np.bool_(True)})
        self.exhausted = False
        self.expire_after_call = False
        self.expire_at_check = None
        self.budget_checks = 0
        self.calls = []
        self.retained_inputs = []
        self.handler = None

    @property
    def budget_exceeded(self):
        self.budget_checks += 1
        return self.exhausted or (self.expire_at_check is not None and self.budget_checks >= self.expire_at_check)

    def value_and_grad_aux(self, params):
        if self.exhausted:
            raise AssertionError("Evaluation after budget exhaustion")
        self.calls.append(params.copy())
        self.retained_inputs.append(params)
        if self.handler is not None:
            return self.handler(params)
        self.exhausted = self.expire_after_call
        return self.result


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.objective = StubObjective()
        self.diagnostics = Diagnostics()
        self.session = EvaluationSession(self.objective, 2, self.diagnostics)
        self.point = np.array([0.5, -0.25])

    def test_exact_repeat_is_cached(self):
        first = self.session(self.point)
        second = self.session(self.point.copy())
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(self.objective.calls), 1)
        self.assertEqual(self.diagnostics.real_evaluations, 1)
        self.assertEqual(self.diagnostics.cache_hits, 1)

    def test_nearby_but_distinct_vector_is_not_cached(self):
        self.session(self.point)
        nearby = self.point.copy()
        nearby[0] = np.nextafter(nearby[0], np.inf)
        self.session(nearby)
        self.assertEqual(len(self.objective.calls), 2)
        self.assertEqual(self.diagnostics.cache_hits, 0)

    def test_cache_is_single_point_not_a_growing_history(self):
        self.session(self.point)
        self.session(self.point + 1.0)
        self.session(self.point)
        self.assertEqual(self.diagnostics.real_evaluations, 3)
        self.assertEqual(self.diagnostics.cache_hits, 0)

    def test_clearing_cache_makes_repeat_a_real_evaluation(self):
        self.session(self.point)
        self.session.clear_cache()
        self.session(self.point)
        self.assertEqual(self.diagnostics.real_evaluations, 2)

    def test_returned_gradient_cannot_poison_cache(self):
        expected = self.objective.result[1].copy()
        _, gradient = self.session(self.point)
        gradient[:] = 99.0
        _, cached = self.session(self.point)
        np.testing.assert_array_equal(cached, expected)
        cached[:] = -99.0
        np.testing.assert_array_equal(self.session(self.point)[1], expected)
        np.testing.assert_array_equal(self.objective.result[1], expected)

    def test_objective_gradient_mutation_cannot_poison_cache(self):
        expected = self.objective.result[1].copy()
        self.session(self.point)
        self.objective.result[1][:] = 999.0
        np.testing.assert_array_equal(self.session(self.point)[1], expected)

    def test_readonly_parameters_and_gradient_are_supported(self):
        self.point.flags.writeable = False
        self.objective.result[1].flags.writeable = False
        loss, gradient = self.session(self.point)
        self.assertEqual(loss, 3.0)
        gradient[0] = -3.0
        np.testing.assert_array_equal(self.point, [0.5, -0.25])
        np.testing.assert_array_equal(self.objective.result[1], [1.0, 2.0])

    def test_objective_receives_owned_input_not_callers_storage(self):
        def mutate_input(params):
            params[:] = 50.0
            return 3.0, np.zeros(2), {"is_feasible": True}

        self.objective.handler = mutate_input
        self.session(self.point)
        np.testing.assert_array_equal(self.point, [0.5, -0.25])
        np.testing.assert_array_equal(self.session.restart_center(np.zeros(2)), self.point)

    def test_mutating_callers_vector_cannot_poison_stored_point(self):
        self.session(self.point)
        self.point[:] = 44.0
        np.testing.assert_array_equal(self.session.restart_center(np.zeros(2)), [0.5, -0.25])
        center = self.session.restart_center(np.zeros(2))
        center[:] = 19.0
        np.testing.assert_array_equal(self.session.restart_center(np.zeros(2)), [0.5, -0.25])

    def test_objective_retained_inputs_are_separate_from_solver_cache_and_restart_arrays(self):
        original = self.point.copy()
        self.session(self.point)
        retained = self.objective.retained_inputs[0]
        self.assertFalse(np.shares_memory(retained, self.point))
        self.point[:] = 77.0
        center = self.session.restart_center(np.zeros(2))
        self.assertFalse(np.shares_memory(retained, center))
        center[:] = -33.0
        self.session(self.point)
        self.session.clear_cache()
        self.session(np.ones(2))
        np.testing.assert_array_equal(retained, original)
        self.assertTrue(all(not np.shares_memory(retained, point) for point in self.objective.retained_inputs[1:]))

    def test_exhausted_budget_prevents_real_call(self):
        self.objective.exhausted = True
        with self.assertRaises(BudgetExhausted):
            self.session(self.point)
        self.assertEqual(self.diagnostics.real_evaluations, 0)

    def test_exhausted_budget_prevents_cached_result(self):
        self.session(self.point)
        self.objective.exhausted = True
        with self.assertRaises(BudgetExhausted):
            self.session(self.point)
        self.assertEqual(self.diagnostics.real_evaluations, 1)
        self.assertEqual(self.diagnostics.cache_hits, 0)

    def test_budget_is_rechecked_immediately_before_cached_return(self):
        self.session(self.point)
        self.objective.expire_at_check = self.objective.budget_checks + 2
        with self.assertRaises(BudgetExhausted):
            self.session(self.point)
        self.assertEqual(self.diagnostics.cache_hits, 0)
        self.assertEqual(len(self.objective.calls), 1)

    def test_budget_is_rechecked_immediately_before_real_call(self):
        self.objective.expire_at_check = 2
        with self.assertRaises(BudgetExhausted):
            self.session(self.point)
        self.assertEqual(len(self.objective.calls), 0)

    def test_final_evaluation_is_observed_before_exiting_scipy(self):
        self.objective.expire_after_call = True
        with self.assertRaises(BudgetExhausted):
            self.session(self.point)
        self.assertEqual(self.diagnostics.real_evaluations, 1)
        np.testing.assert_array_equal(self.session.restart_center(np.zeros(2)), self.point)

    def test_nonfinite_solver_trials_abort_without_evaluation(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad), self.assertRaises(NumericalFailure):
                self.session([bad, 0.0])
        self.assertEqual(self.diagnostics.numerical_failures, 3)
        self.assertEqual(self.diagnostics.real_evaluations, 0)

    def test_nonfinite_loss_is_not_replaced_with_a_finite_value(self):
        for bad in (np.nan, np.inf, -np.inf):
            self.objective.result = (bad, np.zeros(2), {"is_feasible": True})
            with self.subTest(bad=bad), self.assertRaises(NumericalFailure):
                self.session(self.point)
        self.assertEqual(self.diagnostics.numerical_failures, 3)
        np.testing.assert_array_equal(self.session.restart_center(np.zeros(2)), np.zeros(2))

    def test_finite_feasible_loss_is_kept_with_a_nonfinite_gradient(self):
        for bad in (np.nan, np.inf, -np.inf):
            self.objective.result = (4.0, np.array([bad, 0.0]), {"is_feasible": True})
            with self.subTest(bad=bad), self.assertRaises(NumericalFailure):
                self.session(self.point)
            np.testing.assert_array_equal(self.session.restart_center(np.zeros(2)), self.point)
        self.assertEqual(self.diagnostics.real_evaluations, 3)
        self.assertEqual(self.diagnostics.cache_hits, 0)

    def test_best_finite_point_survives_bad_gradient_even_if_infeasible(self):
        self.objective.result = (4.0, np.array([np.nan, 0.0]), {"is_feasible": False})
        with self.assertRaises(NumericalFailure):
            self.session(self.point)
        np.testing.assert_array_equal(self.session.restart_center(np.zeros(2)), self.point)

    def test_feasible_point_is_preferred_over_lower_infeasible_loss(self):
        self.objective.result = (8.0, np.zeros(2), {"is_feasible": True})
        self.session(self.point)
        self.objective.result = (-20.0, np.zeros(2), {"is_feasible": False})
        self.session(np.zeros(2))
        np.testing.assert_array_equal(self.session.restart_center(np.ones(2)), self.point)

    def test_best_feasible_point_uses_returned_loss_not_aux_loss(self):
        self.session(self.point)
        self.objective.result = (
            4.0,
            np.zeros(2),
            {"is_feasible": True, "sensitivity_loss": -1e9, "penalty": -1e9},
        )
        self.session(np.zeros(2))
        np.testing.assert_array_equal(self.session.restart_center(np.ones(2)), self.point)
        self.objective.result = (2.0, np.zeros(2), {"is_feasible": True})
        self.session(np.ones(2))
        np.testing.assert_array_equal(self.session.restart_center(np.zeros(2)), np.ones(2))

    def test_best_finite_is_used_until_a_feasible_point_exists(self):
        self.objective.result = (3.0, np.zeros(2), {"is_feasible": False})
        self.session(self.point)
        self.objective.result = (1.0, np.zeros(2), {"is_feasible": False})
        self.session(np.zeros(2))
        np.testing.assert_array_equal(self.session.restart_center(self.point), np.zeros(2))
        self.objective.result = (9.0, np.zeros(2), {"is_feasible": True})
        self.session(np.ones(2))
        np.testing.assert_array_equal(self.session.restart_center(self.point), np.ones(2))

    def test_ties_keep_the_first_observation(self):
        self.session(self.point)
        self.session(np.zeros(2))
        np.testing.assert_array_equal(self.session.restart_center(np.ones(2)), self.point)

    def test_zero_and_extreme_finite_gradients_are_passed_unchanged(self):
        for gradient in (np.zeros(2), np.array([np.finfo(float).max, -np.finfo(float).max])):
            self.objective.result = (3.0, gradient, {"is_feasible": True})
            self.session.clear_cache()
            with self.subTest(gradient=gradient):
                np.testing.assert_array_equal(self.session(self.point)[1], gradient)
        self.assertEqual(self.diagnostics.numerical_failures, 0)

    def test_bad_gradient_shape_is_an_api_error_not_a_restart(self):
        self.objective.result = (3.0, np.zeros((1, 2)), {"is_feasible": True})
        with self.assertRaisesRegex(ValueError, "Objective gradient must have shape"):
            self.session(self.point)
        self.assertEqual(self.diagnostics.numerical_failures, 0)

    def test_bad_trial_shape_is_an_api_error(self):
        with self.assertRaisesRegex(ValueError, "trial parameters must have shape"):
            self.session(np.zeros((1, 2)))
        self.assertEqual(self.diagnostics.real_evaluations, 0)
        self.assertEqual(self.diagnostics.numerical_failures, 0)

    def test_non_scalar_losses_are_api_errors(self):
        for loss in ([1.0], np.zeros((1, 1))):
            self.objective.result = (loss, np.zeros(2), {"is_feasible": True})
            with self.subTest(loss=loss), self.assertRaisesRegex(ValueError, "loss must be scalar"):
                self.session(self.point)
        self.assertEqual(self.diagnostics.numerical_failures, 0)

    def test_complex_or_text_loss_and_gradient_are_api_errors(self):
        for loss, gradient in ((1j, np.zeros(2)), ("1.0", np.zeros(2)), (3.0, [1j, 2j]), (3.0, ["1", "2"])):
            self.objective.result = (loss, gradient, {"is_feasible": True})
            with self.subTest(loss=loss, gradient=gradient), self.assertRaises(TypeError):
                self.session(self.point)
        self.assertEqual(self.diagnostics.numerical_failures, 0)

    def test_feasibility_must_be_a_boolean_scalar(self):
        for feasible in ([True], 1, 0, np.nan, "False", None):
            self.objective.result = (3.0, np.zeros(2), {"is_feasible": feasible})
            with self.subTest(feasible=feasible), self.assertRaisesRegex(ValueError, "scalar boolean"):
                self.session(self.point)
        self.assertEqual(self.diagnostics.numerical_failures, 0)

    def test_missing_feasibility_is_not_invented(self):
        self.objective.result = (3.0, np.zeros(2), {})
        with self.assertRaises(KeyError):
            self.session(self.point)

    def test_objective_errors_are_not_swallowed(self):
        def broken(_):
            raise RuntimeError("public API failed")

        self.objective.handler = broken
        with self.assertRaisesRegex(RuntimeError, "public API failed"):
            self.session(self.point)
        self.assertEqual(self.diagnostics.numerical_failures, 0)


if __name__ == "__main__":
    unittest.main()
