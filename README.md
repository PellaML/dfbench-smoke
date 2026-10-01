# dfbench-smoke

A small CPU integration check for the public Learn2Design reference optimizers.
It calls the organizer's optimizer and result-summary functions unchanged,
then prints a compact JSON record. It is not a competition entry or an
independent optimization method.

The reference source is pinned to
[Learn2Design-2026 commit 64781a7](https://github.com/artificial-scientist-lab/Learn2Design-2026/tree/64781a778ba546f83104c3692c343b16feb7eaf1).
The Python environment uses dfbench 0.3.3 and JAX 0.9.0.1. The intended runtime is
the Linux test container. Two reference checks have now completed as recorded below.
Configuration and result-summary unit
tests also run on Windows without loading the simulator.

## Run

Build the CPU test image:

```sh
docker build -t dfbench-smoke .
```

Run one full Constrained Voyager problem for a 120-second Objective budget:

```sh
docker run --rm --init --network none --memory 8g --memory-swap 8g \
  --cpus 2 --pids-limit 256 --read-only \
  --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,mode=1777 \
  dfbench-smoke --problem cvoyager --method adam --seconds 120 \
  --seed 42 --require-feasible
```

Use `--problem uifo --seconds 300` for a newly generated UIFO topology. Use
`--method random` for the organizer's uniform-random reference. The seed sets
the optimizer RNG and, for UIFO only, the public topology generator.

The runner uses the dfbench default problem constructors. It does not override
their bounds, frequency count or feasibility criteria. All result-producing calls remain inside the Objective clock.
The reference implementation performs its own permitted warmup before logging.

The default CLI exits successfully when a run completes and a report can be
written. `--require-feasible` adds a stricter check: exit status 3 means no
finite, physically feasible result was logged. A failed feasibility check is a scientific outcome, not necessarily a software
failure. The manual CI workflow keeps this stricter check enabled rather than
turning a run with no feasible point green. Dependency or execution errors return
1, and invalid command-line arguments return 2. A container memory
failure or timeout can stop the process before it produces any report.

## Results

The final line starts with `DFBENCH_RESULT=` and contains JSON, including:

- actual Python/package versions, backend and x64 setting;
- problem, method, seed and Objective budget;
- total probe time, optimizer wall time including warmup, and the separate
  Objective clock;
- logged evaluation count, finite/feasible candidate counts and missing aux data;
- best feasible loss, or null when there is none;
- Linux process peak RSS, measured inside the interpreter.

The feasibility summary is the organizer's own `summarize_objective` function.
It handles scalar and batched histories without treating missing feasibility
information as a valid result. A lower finite loss alone does not establish
physical feasibility. No organizer fallback score is invented here.

JIT warmup is not timed separately. The difference between wall time and the
Objective clock must not be presented as an isolated compilation measurement.
Workflow step timings can be used to report the actual image-build duration.

A passing smoke means only that this public case ran and met the selected
check. It does not predict the private leaderboard, prove convergence or
represent a four-hour H100 run. No competitive result is claimed.

## Verified reference runs

The two checks below completed on 1 October 2026 at code commit
[471a451](https://github.com/PellaML/dfbench-smoke/commit/471a45114402bffa474607cce102056628b4616a).
Both used the unchanged organizer Adam method, optimizer seed 42, Linux CPU
and the bounded CI container. The UIFO topology seed was also 42.

| Public case | Parameters | Evaluations | Feasible candidates | Best feasible loss | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Constrained Voyager](https://github.com/PellaML/dfbench-smoke/actions/runs/36893019389) | 48 | 638 | 36 | 4.727176 | 1.92 GiB |
| [UIFO](https://github.com/PellaML/dfbench-smoke/actions/runs/36894476101) | 187 | 71 | 40 | 4.600065 | 6.80 GiB |

The configured Objective budgets were 120 and 300 seconds. The observed
Objective clocks were 120.14 and 303.75 seconds; these are reported as
observed rather than rounded down to the budgets. Total probe times were
164.46 and 335.45 seconds. The full machine-readable records and package
versions are in [docs/reference-runs.json](docs/reference-runs.json).

These are different problems, not a before/after performance comparison.
They establish that these two public cases executed and produced feasible
results. They do not establish generalization across topologies, an
improvement over the reference method, or a private H100 competition score.

## CI limits

The workflow is manual-only. It uses a standard public `ubuntu-24.04` runner,
a 20-minute job limit, and at most one active run. The scientific process is
inside an 8 GiB, two-CPU container with no network and a read-only root
filesystem. Temporary compiler/cache files stay in bounded `/tmp` storage.
OpenBLAS/OpenMP/MKL thread hints are set to one; those hints do not control
every XLA thread. Docker's CPU quota is a separate limit.

Unit tests use a separate 1 GiB container. The scientific container is removed
on exit, including failures. No Actions cache, uploaded artifact, image push,
GPU, larger runner or paid API is configured. Results remain in workflow logs.

This workflow tests only this repository's software. It is not a generic
hosted-compute endpoint or a substitute for the official evaluator.

## Local unit tests

With NumPy available, these tests do not import JAX or run a simulation:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Scientific dependencies are pinned with compatible wheel hashes in
`requirements-linux.lock`, targeting CPython 3.12 on Linux x86-64 with glibc
2.36. The Docker base image and checkout action are also pinned by digest or
commit. The C++ runtime comes from that base image and is checked during the
build, rather than installed from an unpinned package index. Vendored-file
hashes are checked by the test suite.

## Source notices

`src/dfbench_smoke/vendor` contains unmodified organizer code. Its original
paths and hashes are listed in `UPSTREAM.json`; the upstream MIT notice is in
`third_party/learn2design-MIT.txt`. See `NOTICE` for attribution. These are upstream reference methods, not new
algorithms developed in this project.
