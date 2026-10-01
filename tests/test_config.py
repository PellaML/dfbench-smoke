import unittest

from dfbench_smoke.config import ProbeConfig


class ConfigTests(unittest.TestCase):
    def test_defaults_are_a_bounded_public_smoke(self):
        self.assertEqual(ProbeConfig(), ProbeConfig("cvoyager", "adam", 120, 42))

    def test_budget_boundaries(self):
        for value in (1, 1.5, 300):
            with self.subTest(value=value):
                self.assertEqual(ProbeConfig(seconds=value).seconds, value)

    def test_bad_budgets_are_rejected(self):
        for value in (0, -1, 301, float("inf"), float("nan"), True, "120", None, 10**1000):
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                ProbeConfig(seconds=value)

    def test_seed_boundaries(self):
        for seed in (0, 2**31 - 1):
            self.assertEqual(ProbeConfig(seed=seed).seed, seed)
        for seed in (-1, 2**31, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                ProbeConfig(seed=seed)

    def test_unknown_problem_and_method(self):
        with self.assertRaises(ValueError):
            ProbeConfig(problem="private-topology")
        with self.assertRaises(ValueError):
            ProbeConfig(method="shell-command")
