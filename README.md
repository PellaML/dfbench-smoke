# dfbench-smoke

Bounded CPU checks for Learn2Design reference methods and an experimental hybrid
optimizer. The `adam` and `random` methods remain unchanged upstream code. The
`hybrid` method is a separate engineering candidate using standard numerical
components. No competition entry or performance advantage is claimed.

The reference source is pinned to
[Learn2Design-2026 commit 64781a7](https://github.com/artificial-scientist-lab/Learn2Design-2026/tree/64781a778ba546f83104c3692c343b16feb7eaf1).
The Python environment uses dfbench 0.3.3 and JAX 0.9.0.1. The intended runtime is
the Linux test container. Two reference checks and one same-host paired UIFO check
have completed as recorded below. Configuration and result-summary unit tests also
run on Windows without loading the simulator.

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
`--method random` for the organizer's uniform-random reference or `--method hybrid`
for the candidate. The seed sets the optimizer RNG and, for UIFO only, the public
topology generator.

The runner uses the dfbench default problem constructors. It does not override
their bounds, frequency count or feasibility criteria. Initial parameter selection
and Objective warmup follow the allowed pre-logging setup. Every search-guiding
Objective evaluation goes through the logged public API.

The default CLI exits successfully when a run completes and a report can be
written. `--require-feasible` checks the recorded-time window: exit status 3 means
no finite, physically feasible result has an Objective timestamp in that window.
If those timestamps cannot be verified, the command exits 1 instead. A failed
feasibility check is a scientific outcome, not necessarily a software failure. The manual CI workflow keeps this stricter check enabled rather than
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
- canonical best feasible loss, or null when there is none;
- a separate recorded-time-window summary and any unavailable-data reason;
- Linux process peak RSS, measured inside the interpreter;
- candidate work counters when available, not a second score or evaluation count.

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

## Hybrid candidate

The candidate uses a short Adam warm start, then SciPy L-BFGS-B local solves in
the same unbounded coordinates. Its warm phase uses the public AdamGD reference
settings, reimplemented in NumPy rather than calling the vendored update:
learning rate 0.1, moments 0.9/0.999, epsilon 1e-8 and global-norm clipping at 1.
It starts at most twelve warm attempts, uses a floor-rounded 15%
evaluation allowance when one is configured, and checks the tightest budget's
15% progress threshold before each attempt. A single call is not preemptible.

Local solves use true, unclipped derivatives, with L-BFGS history 10,
`maxls=20`, `gtol=1e-6` and `ftol=1e-12`. Normal SciPy convergence safeguards
remain under the unchanged Objective budget. A solver-limit exit continues
from an observed point without random perturbation. Converged, abnormal or
numerically failed solves use two local Gaussian restarts (scales 0.3 and 0.8),
then a new Objective random sample. The center prefers the best observed
feasible point, falling back to the best finite point. No private topology,
score, penalty, physical bound or logging clock is changed.

Exact repeated solver requests can use a solve-local cache. Every actual query
uses `value_and_grad_aux` and checks the public budget before and after the
call. Parameter copies retained by the Objective are not reused as mutable
solver or cache storage. Counters describe calls, updates and solve outcomes;
they can differ from admitted Objective history entries.

This combines established Adam and L-BFGS methods; it is not a claim of a new
mathematical algorithm. In the first paired public UIFO check below, the candidate
was worse than Adam. Analytic tests do not establish a UIFO advantage.

## Recorded-time accounting

Schema version 2 keeps the organizer's canonical summary and adds a second
view using the public `time_steps` and `max_time` properties. It follows the
closed `[0, T]` interval in the scoring document. Late records are excluded,
while missing, invalid or nonmonotonic timestamps make that view unavailable.
The Objective is not modified and no replacement score is written back to it.

These are the Objective's own wall-clock readings. They may be recorded before
all computation finishes and are not independent completion timestamps. A clock
step backwards can also make the diagnostic unavailable. Do not interpret the
window as proof of an exact external wall-time cutoff.

The earlier stored reference runs below predate this diagnostic and contain no
per-evaluation timestamps. A candidate comparison therefore needs a fresh
baseline at the same code revision, not a reconstructed score for those runs.

## First paired UIFO check

The [paired run](https://github.com/PellaML/dfbench-smoke/actions/runs/36917528550)
completed on 1 October 2026 at code commit
[02c0904](https://github.com/PellaML/dfbench-smoke/commit/02c0904abfda9bc9291503aeee77fb060455f2c4).
Adam and the hybrid ran in that order in fresh containers on the same host and
image. Both used public topology seed 42, optimizer seed 42, 187 parameters,
a configured 300-second Objective budget, a two-CPU quota and an 8 GiB
memory/swap limit. All 166 unit and provenance tests passed.

| Method | Evaluations | Feasible observations | Best feasible loss | Objective clock at reporting | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adam reference | 42 | 24 | 4.815239 | 302.38 s | 6.73 GiB |
| Hybrid candidate | 41 | 22 | 4.909185 | 304.21 s | 6.72 GiB |

Lower loss is better. The candidate was worse by 0.093946 on this case; the green
workflow means both produced a feasible result, not that the candidate improved
on the reference. Each container exited 0, was not OOM-killed or timed out, and
was removed successfully.

For both methods, the canonical and recorded-time-window best losses match.
The inspected dfbench 0.3.3 implementation admits records only when its sampled
elapsed time is below the budget. Zero excluded records are therefore expected
for these fresh runs, not independent confirmation of an exact completion-time
cutoff. Recorded timestamps may precede output synchronization.

The Objective clock keeps running after the optimizer returns and is sampled
when the report is built. Its 302.38 and 304.21 second readings include work up
to that point; they are not exact optimizer durations. This is a separate issue
from the timing of individual records. Process RSS is not peak container usage
or a memory bound for longer runs.

The hybrid completed five warm Adam steps and entered one L-BFGS-B solve before
the budget stopped it. It recorded no restarts, cache hits or numerical failures.
The run has no per-evaluation trajectory, so those counters do not establish why
it lost or support a matched-evaluation comparison. Optimizer wall times were
330.07 and 331.44 seconds; total probe times were 350.64 and 351.82 seconds for
Adam and the hybrid respectively. These include work outside the logged budget.

The earlier unchanged Adam check below used the same seed and budget but
admitted 71 evaluations and reached 4.600065, versus 42 and 4.815239 here.
Throughput cannot be assumed constant across CI jobs; those two historical
outcomes do not estimate run-to-run variance. The paired order was not reversed,
and numerical agreement between the two warm-start paths was not traced.

This is one public topology and one seed, without statistical replication. It
does not establish that the observed gap exceeds run-to-run variation, nor does
it establish generalization, a private H100 score or prize eligibility.
Full result and controller records are in
[docs/paired-uifo-run.json](docs/paired-uifo-run.json). The older runs below are
historical checks, not additional pairs in this comparison.

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
Objective clocks at reporting were 120.14 and 303.75 seconds; these are reported as
observed rather than rounded down to the budgets. Total probe times were
164.46 and 335.45 seconds. The full machine-readable records and package
versions are in [docs/reference-runs.json](docs/reference-runs.json).

These are different problems, not a before/after performance comparison.
They establish that these two public cases executed and produced feasible
results. They do not establish generalization across topologies, an
improvement over the reference method, or a private H100 competition score.

## CI limits

The workflow is manual-only. It uses a standard public `ubuntu-24.04` runner,
a 20-minute job limit, and at most one active job. Normal modes run one case.
`compare` runs exactly the reference Adam and the hybrid candidate sequentially
on that same host, in separate fresh containers with the same public problem,
seed and budget. It is not an input for arbitrary parameter sweeps. The scientific process is
inside an 8 GiB, two-CPU container with no network and a read-only root
filesystem. Temporary compiler/cache files stay in bounded `/tmp` storage.
OpenBLAS/OpenMP/MKL thread hints are set to one; those hints do not control
every XLA thread. Docker's CPU quota is a separate limit.

Unit tests use a separate 1 GiB container. Each scientific case has an eight-minute
outer wall limit. A disconnected observer waits for the same container rather
than restarting it. Cleanup is checked for each confirmed container; the
controller stops before another case if removal is unconfirmed. No Actions cache, uploaded artifact, image push,
GPU, larger runner or paid API is configured. Results remain in workflow logs.

This workflow tests only this repository's software. It is not a generic
hosted-compute endpoint or a substitute for the official evaluator.

## Local unit tests

With NumPy and SciPy available, these tests do not import JAX or run a simulation:

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
