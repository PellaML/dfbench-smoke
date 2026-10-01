"""Run at most two bounded optimizer checks in separate containers on one CI host."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, replace

from dfbench_smoke.config import ProbeConfig

_WALL_SECONDS = 480
_CONTROL_SECONDS = 30
_BUDGETS = {"cvoyager": 120, "uifo": 300}


@dataclass
class CaseResult:
    method: str
    problem: str
    passed: bool = False
    timeout: bool = False
    container_exit_code: int | None = None
    oom_killed: bool | None = None
    cleanup_confirmed: bool = False
    recovered_observation: bool = False
    control_error: str | None = None
    elapsed_seconds: float | None = None


class Docker:
    def call(self, *arguments: str, capture: bool = True, timeout: float = _CONTROL_SECONDS):
        return subprocess.run(
            ["docker", *arguments],
            capture_output=capture,
            text=True,
            timeout=timeout,
            check=False,
        )

    def state(self, container: str) -> dict:
        response = self.call("inspect", "--format", "{{json .State}}", container)
        if response.returncode:
            raise RuntimeError("Cannot inspect the owned test container.")
        state = json.loads(response.stdout)
        if (
            not isinstance(state, dict)
            or not isinstance(state.get("Running"), bool)
            or not isinstance(state.get("OOMKilled"), bool)
            or isinstance(state.get("ExitCode"), bool)
            or not isinstance(state.get("ExitCode"), int)
            or not isinstance(state.get("Status"), str)
        ):
            raise ValueError("Docker returned an invalid state record.")
        return state


def create_arguments(image: str, config: ProbeConfig) -> list[str]:
    if not re.fullmatch(r"dfbench-smoke:[0-9a-f]{40}", image):
        raise ValueError("Use the local test image tagged with the current full commit hash.")
    return [
        "create",
        "--init",
        "--network",
        "none",
        "--memory",
        "8g",
        "--memory-swap",
        "8g",
        "--cpus",
        "2",
        "--pids-limit",
        "256",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
        image,
        "--problem",
        config.problem,
        "--method",
        config.method,
        "--seconds",
        str(config.seconds),
        "--seed",
        str(config.seed),
        "--require-feasible",
    ]


def _stop_running(docker: Docker, container: str) -> None:
    docker.call("kill", container)
    state = docker.state(container)
    if state["Running"]:
        raise RuntimeError("The owned test container is still running after its time limit.")


def run_case(docker: Docker, image: str, config: ProbeConfig) -> CaseResult:
    result = CaseResult(config.method, config.problem)
    response = docker.call(*create_arguments(image, config))
    container = response.stdout.strip() if response.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise RuntimeError("Container creation was not confirmed; no automatic retry is allowed.")
    started = time.monotonic()
    try:
        try:
            attached = docker.call("start", "--attach", container, capture=False, timeout=_WALL_SECONDS)
        except subprocess.TimeoutExpired:
            result.timeout = True
            _stop_running(docker, container)
        else:
            state = docker.state(container)
            # A detached or failed observer is not a completed workload. Wait
            # for the same container, never create a replacement for it.
            if state["Running"]:
                remaining = _WALL_SECONDS - (time.monotonic() - started)
                if remaining <= 0:
                    result.timeout = True
                    _stop_running(docker, container)
                else:
                    try:
                        waited = docker.call("wait", container, timeout=remaining)
                        if waited.returncode:
                            raise RuntimeError("Could not observe the existing container to completion.")
                        result.recovered_observation = True
                        logs = docker.call("logs", container, capture=False)
                        if logs.returncode:
                            raise RuntimeError("Completed-container logs could not be retrieved.")
                    except subprocess.TimeoutExpired:
                        result.timeout = True
                        _stop_running(docker, container)
            elif attached.returncode not in (0, state.get("ExitCode")):
                logs = docker.call("logs", container, capture=False)
                if logs.returncode:
                    raise RuntimeError("Completed-container logs could not be retrieved.")
                result.recovered_observation = True
        state = docker.state(container)
        if state["Running"] or state.get("Status") != "exited":
            raise RuntimeError("No terminal exit state is available for this container.")
        result.container_exit_code = int(state["ExitCode"])
        result.oom_killed = state["OOMKilled"]
        result.passed = not result.timeout and result.container_exit_code == 0 and not result.oom_killed
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result.control_error = str(error)
    finally:
        try:
            removed = docker.call("rm", "--force", container)
            result.cleanup_confirmed = removed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            result.cleanup_confirmed = False
        result.passed = result.passed and result.cleanup_confirmed and result.control_error is None
        result.elapsed_seconds = time.monotonic() - started
    return result


def configurations(problem: str, method: str, seed: int) -> list[ProbeConfig]:
    methods = ("adam", "hybrid") if method == "compare" else (method,)
    config = ProbeConfig(problem, methods[0], _BUDGETS[problem], seed)
    return [replace(config, method=selected) for selected in methods]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--problem", choices=tuple(_BUDGETS), required=True)
    parser.add_argument("--method", choices=("adam", "random", "hybrid", "compare"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    docker = Docker()
    passed = True
    for config in configurations(args.problem, args.method, args.seed):
        result = run_case(docker, args.image, config)
        print("DFBENCH_CASE_STATUS=" + json.dumps(vars(result), sort_keys=True), flush=True)
        passed = passed and result.passed
        if not result.cleanup_confirmed:
            # Do not risk overlapping workloads when cleanup is unconfirmed.
            break
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
