import contextlib
import io
import json
import unittest
from unittest.mock import patch

from dfbench_smoke.cli import main
from dfbench_smoke.config import ProbeConfig
from dfbench_smoke.probe import run_probe


class CliTests(unittest.TestCase):
    def invoke(self, argv, result=None, error=None):
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("dfbench_smoke.cli.run_probe", return_value=result, side_effect=error) as run,
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            status = main(["--problem", "cvoyager", *argv])
        return status, output.getvalue(), errors.getvalue(), run

    def test_result_has_a_parseable_marker(self):
        status, output, errors, run = self.invoke(
            ["--seed", "9"], {"feasible_candidate_count": 1, "best_feasible_loss": -3.5}
        )
        self.assertEqual(status, 0)
        self.assertEqual(errors, "")
        self.assertEqual(json.loads(output.removeprefix("DFBENCH_RESULT="))["best_feasible_loss"], -3.5)
        self.assertEqual(run.call_args.args[0].seed, 9)

    def test_missing_feasible_result_is_not_a_success_when_required(self):
        status, output, errors, _ = self.invoke(
            ["--require-feasible"],
            {
                "feasible_candidate_count": 0,
                "best_feasible_loss": None,
                "time_window_status": "complete",
                "time_window_feasible_candidate_count": 0,
            },
        )
        self.assertEqual(status, 3)
        self.assertIn('"best_feasible_loss": null', output)
        self.assertIn("no finite physically feasible result", errors)

    def test_unconstrained_smoke_can_report_no_feasible_result(self):
        status, _, _, _ = self.invoke([], {"feasible_candidate_count": 0})
        self.assertEqual(status, 0)

    def test_errors_are_readable(self):
        status, output, errors, _ = self.invoke([], error=RuntimeError("missing test dependency"))
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("missing test dependency", errors)

    def test_nonfinite_report_is_rejected(self):
        status, output, _, _ = self.invoke([], {"feasible_candidate_count": 0, "bad": float("nan")})
        self.assertEqual(status, 1)
        self.assertEqual(output, "")

    def test_bad_budget_never_runs_a_simulator(self):
        with (
            patch("dfbench_smoke.cli.run_probe") as run,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["--problem", "cvoyager", "--seconds", "inf"])
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()

    def test_windows_execution_stops_before_scientific_imports(self):
        with (
            patch("dfbench_smoke.probe.platform.system", return_value="Windows"),
            self.assertRaisesRegex(RuntimeError, "Linux test container"),
        ):
            run_probe(ProbeConfig())

    def test_no_arguments_never_start_a_simulator(self):
        with (
            patch("dfbench_smoke.cli.run_probe") as run,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main([])
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()

    def test_late_only_feasibility_does_not_pass_the_gate(self):
        status, _, errors, _ = self.invoke(
            ["--require-feasible"],
            {
                "feasible_candidate_count": 1,
                "time_window_status": "complete",
                "time_window_feasible_candidate_count": 0,
            },
        )
        self.assertEqual(status, 3)
        self.assertIn("within budget", errors)

    def test_missing_timestamps_do_not_pass_the_gate(self):
        status, _, errors, _ = self.invoke(
            ["--require-feasible"],
            {
                "feasible_candidate_count": 1,
                "time_window_status": "unavailable",
                "time_window_feasible_candidate_count": None,
            },
        )
        self.assertEqual(status, 1)
        self.assertIn("could not be verified", errors)

    def test_timed_feasible_result_passes_the_gate(self):
        status, _, _, _ = self.invoke(
            ["--require-feasible"],
            {
                "feasible_candidate_count": 1,
                "time_window_status": "complete",
                "time_window_feasible_candidate_count": 1,
            },
        )
        self.assertEqual(status, 0)
