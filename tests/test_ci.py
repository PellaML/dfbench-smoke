import contextlib
import io
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dfbench_smoke.ci import Docker, configurations, create_arguments, main, run_case
from dfbench_smoke.config import ProbeConfig

IMAGE = "dfbench-smoke:" + "a" * 40
CONTAINER = "b" * 64


class FakeDocker(Docker):
    def __init__(self, exit_code=0, *, timeout=False, detach=False, remove_ok=True, malformed=False, create_ok=True):
        self.calls = []
        self.exit_code = exit_code
        self.timeout = timeout
        self.detach = detach
        self.remove_ok = remove_ok
        self.malformed = malformed
        self.create_ok = create_ok
        self.running = False
        self.killed = False

    def call(self, *arguments, capture=True, timeout=30):
        self.calls.append((arguments, capture, timeout))
        command = arguments[0]
        response = SimpleNamespace(returncode=0, stdout="", stderr="")
        if command == "create":
            response.returncode = 0 if self.create_ok else 1
            response.stdout = CONTAINER if self.create_ok else ""
        elif command == "start":
            self.running = True
            if self.timeout:
                raise subprocess.TimeoutExpired(arguments, timeout)
            if self.detach:
                response.returncode = 125
            else:
                self.running = False
                response.returncode = self.exit_code
        elif command == "wait":
            self.running = False
            response.stdout = str(self.exit_code)
        elif command == "kill":
            self.running = False
            self.killed = True
            self.exit_code = 137
        elif command == "inspect":
            state = {
                "Running": self.running,
                "Status": "running" if self.running else "exited",
                "ExitCode": self.exit_code,
                "OOMKilled": False,
            }
            if self.malformed:
                del state["OOMKilled"]
            response.stdout = json.dumps(state)
        elif command == "rm":
            response.returncode = 0 if self.remove_ok else 1
        elif command != "logs":
            raise AssertionError(f"Unexpected Docker call: {command}")
        return response


class ControllerTests(unittest.TestCase):
    def test_comparison_is_two_fixed_methods_with_matching_conditions(self):
        configs = configurations("uifo", "compare", 42)
        self.assertEqual([c.method for c in configs], ["adam", "hybrid"])
        self.assertEqual({c.seconds for c in configs}, {300})
        self.assertEqual({c.seed for c in configs}, {42})
        self.assertEqual({c.problem for c in configs}, {"uifo"})

    def test_normal_mode_is_one_case(self):
        self.assertEqual(configurations("cvoyager", "random", 7), [ProbeConfig("cvoyager", "random", 120, 7)])

    def test_creation_arguments_keep_security_and_resource_limits(self):
        args = create_arguments(IMAGE, ProbeConfig("uifo", "hybrid", 300, 42))
        for flag, value in (
            ("--network", "none"),
            ("--memory", "8g"),
            ("--memory-swap", "8g"),
            ("--cpus", "2"),
            ("--pids-limit", "256"),
            ("--cap-drop", "ALL"),
            ("--security-opt", "no-new-privileges"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--read-only", args)
        self.assertIn("--require-feasible", args)
        self.assertNotIn("--privileged", args)

    def test_arbitrary_images_are_rejected_before_any_runtime(self):
        for image in ("external/image", "--privileged", "dfbench-smoke:latest", "dfbench-smoke:" + "g" * 40):
            with self.subTest(image=image), self.assertRaises(ValueError):
                create_arguments(image, ProbeConfig())

    def test_success_requires_terminal_state_and_cleanup(self):
        docker = FakeDocker()
        result = run_case(docker, IMAGE, ProbeConfig())
        self.assertTrue(result.passed)
        self.assertTrue(result.cleanup_confirmed)
        self.assertEqual(result.container_exit_code, 0)
        self.assertFalse(result.oom_killed)
        self.assertEqual([c[0][0] for c in docker.calls].count("create"), 1)

    def test_scientific_failure_is_not_hidden(self):
        result = run_case(FakeDocker(exit_code=3), IMAGE, ProbeConfig())
        self.assertFalse(result.passed)
        self.assertEqual(result.container_exit_code, 3)
        self.assertTrue(result.cleanup_confirmed)

    def test_timeout_stops_only_the_existing_container(self):
        docker = FakeDocker(timeout=True)
        result = run_case(docker, IMAGE, ProbeConfig())
        self.assertTrue(result.timeout)
        self.assertTrue(docker.killed)
        self.assertTrue(result.cleanup_confirmed)
        self.assertFalse(result.passed)
        self.assertEqual([c[0][0] for c in docker.calls].count("create"), 1)

    def test_detached_observer_waits_for_same_container_without_restart(self):
        docker = FakeDocker(detach=True)
        result = run_case(docker, IMAGE, ProbeConfig())
        self.assertTrue(result.passed)
        self.assertTrue(result.recovered_observation)
        commands = [c[0][0] for c in docker.calls]
        self.assertEqual(commands.count("create"), 1)
        self.assertEqual(commands.count("start"), 1)
        self.assertEqual(commands.count("wait"), 1)
        self.assertIn("logs", commands)

    def test_failed_cleanup_cannot_pass(self):
        result = run_case(FakeDocker(remove_ok=False), IMAGE, ProbeConfig())
        self.assertFalse(result.passed)
        self.assertFalse(result.cleanup_confirmed)

    def test_missing_state_fields_are_unknown_not_false(self):
        result = run_case(FakeDocker(malformed=True), IMAGE, ProbeConfig())
        self.assertFalse(result.passed)
        self.assertIsNone(result.oom_killed)
        self.assertIsNotNone(result.control_error)
        self.assertTrue(result.cleanup_confirmed)

    def test_unconfirmed_creation_is_not_retried(self):
        docker = FakeDocker(create_ok=False)
        with self.assertRaisesRegex(RuntimeError, "not confirmed"):
            run_case(docker, IMAGE, ProbeConfig())
        self.assertEqual(len(docker.calls), 1)

    def test_comparison_stops_before_another_case_if_cleanup_is_unknown(self):
        docker = FakeDocker(remove_ok=False)
        with (
            patch("dfbench_smoke.ci.Docker", return_value=docker),
            patch("sys.argv", ["ci", "--image", IMAGE, "--problem", "uifo", "--method", "compare"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = main()
        self.assertEqual(status, 1)
        self.assertEqual([call[0][0] for call in docker.calls].count("create"), 1)

    def test_scientific_failure_does_not_skip_the_second_comparison_case(self):
        docker = FakeDocker(exit_code=3)
        output = io.StringIO()
        with (
            patch("dfbench_smoke.ci.Docker", return_value=docker),
            patch("sys.argv", ["ci", "--image", IMAGE, "--problem", "uifo", "--method", "compare"]),
            contextlib.redirect_stdout(output),
        ):
            status = main()
        self.assertEqual(status, 1)
        self.assertEqual([call[0][0] for call in docker.calls].count("create"), 2)
        results = [json.loads(line.split("=", 1)[1]) for line in output.getvalue().splitlines()]
        self.assertEqual([item["method"] for item in results], ["adam", "hybrid"])
        self.assertTrue(all(item["cleanup_confirmed"] for item in results))
