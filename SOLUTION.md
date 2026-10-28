# Direct Emmy Kernel IR to PTX

This submission implements its own typed IR traversal and PTX instruction emission
in [ptx_backend.py](ptx_backend.py). Emmy **0.3.11** supplies deserialization and IR
definitions. No existing CUDA/PTX compiler or Emmy renderer performs the translation.
The generated PTX 8.0 normally targets `sm_89` and selects `sm_90` for bulk-copy/barrier nodes; the runner's `ptxas` assembles it downstream.

Run these commands from the repository root:

```console
python -m pip install -r requirements.txt
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

This compiles the first public problem. In the general interface
`python compile.py kernel.ir.json > out.ptx`, `kernel.ir.json` is a placeholder
for an existing IR file, not a file shipped at the repository root. For example,
compile a reduction with:

```console
python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > sum.ptx
```

In Windows PowerShell 5.1, use `cmd /c` for redirection so the PTX file is not
converted to UTF-16:

```powershell
cmd /c "python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx"
```

Use Python 3.12 or newer and keep the supplied `meta.json` beside the IR. Its
argument order, shared-memory allocation, and numerical tolerances form the launch
contract. Compilation needs no GPU or network. Stdout contains PTX; errors use
stderr and a nonzero status. The submission consists of `compile.py`,
`ptx_backend.py`, `ptx_tensor.py`, `ptx_bulk.py`, and `requirements.txt`.

## Current manual-review release

The current [phase-8 revision](docs/PHASE8.md) emits launch bounds from the
prescribed block size and corrects integer loop bindings and floating floor
remainder. The diagnostic harness now compares zero-tolerance integer outputs
exactly, preserving large integer failure values. A fresh run through the
unchanged organizer client passed **15/15 public problems**, and the final
additional feature gate passed **51/51 packages**. CPU/assembler discovery
passed **120 methods**, with 19 explicit local-GPU skips out of 139 methods. The
[source-bound summary](artifacts/validation/phase8/release-summary.json) records
the current feature, CPU/assembler and checkout evidence; these outcomes are
distinct from the historical counts below.

The phase-8 feature gate covers 41 retained packages, six new loop-binding
packages and four new remainder packages. Its initial **49/51 run remains a
failed gate**: the two FP64 raw-I64 fixtures lost low bits in runner transport.
The final roster substitutes two-I32-word fixtures with unchanged references
and zero tolerance; the failed attempts are retained. The
[phase-8 report](docs/PHASE8.md) records this distinction and the independent
word-transport audit. All GPU validation for this revision uses the organizers'
runner. The [phase-7 report](docs/PHASE7.md), including its 41 additional GPU
passes and 110 CPU/assembler passes, remains evidence for the previous compiler.

The [submission overview](SUBMISSION.md) and [feature index](docs/FEATURES.md)
cover the tensor-core and bulk-copy extensions, examples, tests and exact limits.
Use this source checkout for the repository submission; a new ZIP is not needed.
The [checkout notes](docs/CHECKOUT.md) explain the version-2 inventory, current
runtime references and verification commands. The preserved
[phase-7 archive pointers](artifacts/release/phase7/review/latest.json) describe
the previous packaged release, not the current backend. The
[phase-6 report](docs/PHASE6.md) remains the historical snapshot described in
the recorded video; later reports document the subsequent corrections.
Updating this checkout does not itself push the public repository or update the entry.

## Historical phase-4 runtime artifact

The prior phase 4 submission archive (`submission-bcbc4a8381497f1d34be.zip`, separate historical release artifact)
contains the original three runtime files. Its
[phase-4 manifest](artifacts/release/phase4/submission-bcbc4a8381497f1d34be-3908fd80497591da4d12.manifest.json)
binds the unchanged source hashes and six evidence snapshots, including the
24 October review with its retained precision-incompatible failures; the archive SHA-256 is
`b9af6a761233e39b8faa330e1fd01c49e8a457498cb5d3b7e2465ef5403251e8`.
The [phase-4 PTX manifest](build/open-ptx-phase4-cli/manifest.json) links all 15
actual CLI outputs, each in its own problem directory. Their bytes match the
official runner payloads in the
[15/15 identity check](artifacts/validation/phase4/public-ptx-identity.json).
This export supersedes `build/open-ptx-phase4`, whose filename collisions and
newline handling made it unsuitable as an exact CLI export. Generate `out.ptx`
with the command above; the compiler sources are the submitted program.

## Previous scalar-only runner review (24 October 2026)

All **15/15 public problems pass** using the unmodified
`python submit.py <problem> --compiler compile.py` workflow. Passing the public
set is the baseline. The [fresh review](docs/FINAL_CHECK_2026_10_24.md) adds real
FP16, FP64 and I32 storage, ragged shapes and different operations through the
repository generator and separately labelled typed fixtures.

| Fresh executed checks | Mathematical-reference result |
|---|---:|
| Public client rerun | 15/15 pass |
| 16 generated graphs over three inputs, plus one large-I32 input variant | 40/49 pass; 9 retained failures |
| Two hand-built operation fixtures over three inputs | 6/6 pass |
| Three explicit FP64-state IR variants over three inputs | 9/9 pass |
| Total | 70 pass, 9 retained failures out of 79 |

The nine failures belong to generated FP64-buffer sum, matmul and normalization
packages whose scheduled arithmetic/state is FP32. Their returned values match
independent explicit-FP32 IR oracles bit-for-bit. For 3,052 of 3,057 tested
output elements, even the closest FP32 value cannot satisfy the original FP64
mathematical tolerance. The three FP64-state variants change the **test IR**;
they pass the unchanged original tolerances and expected arrays. They are not
compiler repairs or unchanged generator outputs, and they do not erase the
original failures.

Thus the new outside-public set records **55 passing and 9 failing mathematical
checks out of 64**. The [final review index](artifacts/validation/2026-10-24-final/summary.json)
records both results and their attribution. Its review acceptance is not an
all-passing mathematical gate. At that review the production compiler and ZIP
were unchanged; phase 5 subsequently added the explicit hardware extensions. The older 332-check snapshot below remains historical evidence;
repeated public checks are not added to it as new robustness coverage.

## Coverage and design

The generic lowerer handles all five competition families: pointwise operations,
reductions, matrix multiplication, fused normalization/softmax, and attention-like
kernels. It translates statement structure and dataflow, without problem-name
dispatch, reference-code reads, expected-output access, or precomputed answers.

- Loop seeding and `Accum.base` preserve nested and rescaled recurrence state;
  immutable snapshots remain valid after a mutable accumulator changes.
- Warp and shared-memory reductions translate the ordered state-combination
  program, preserve subgroup participation, and broadcast results to consumers.
- Barriers retain their identifiers and participant counts. Asynchronous copies,
  commit groups, waits, and subsequent consumer synchronization remain distinct.
- Addressing follows tensor layouts and prescribed thread mapping. Vector memory
  operations require proven alignment; split-K writes use atomic accumulation.
- Loop variables use signed 32-bit bindings. Explicit ends are captured once
  after initialization; live increments compute with promoted operands before
  assigning back to the induction variable.
- Named FP32/FP64 floor remainder uses exact power-of-two scaling and subtraction
  instead of forming a rounded quotient. Integer-bit transport fixtures check
  low-bit results, subnormals, signed zero and canonicalized NaN outcomes.

The continued audit also fixed four general cases: capture mutable strided-loop
bounds once at entry; implement the source IR's legacy flattened-index convention;
promote predicate operands for integer arithmetic; and preserve valid alignment
facts across fixed-width integer overflow. Each has an independent regression.

Phase 4 adds explicit operation/type/arity admission, per-expression FP16/BF16
rounding, target-aware BF16 conversion, and legal small-integer selection and
bitcasts. Unsupported combinations fail with a diagnostic and no partial PTX.
[TYPE_SUPPORT.md](docs/TYPE_SUPPORT.md) states the exact boundaries, including
the deliberate NumPy magnitude semantics for named signed `bitwise_count` and
its disagreement with the pinned CUDA renderer. This is not a claim to implement
every NumPy operation or a correctly rounded transcendental library.

Independent interaction fixtures cover guarded invalid addresses with live
state, ragged rows whose alignment changes between rows, scalar cleanup, and
a two-slot asynchronous pipeline with seven stages, cross-warp consumers and
a final drain. [Transformation contracts](docs/TRANSFORMATION_CONTRACTS.md)
record optimization preconditions. The development analyzer reports structural
features and optional assembler resource data; it is not imported by the
submitted compiler and does not prove arbitrary memory safety or coalescing.

## Numerical and performance choices

Ordinary arithmetic uses explicit rounding. For independent additive dot products,
nonzero metadata tolerances enable compensated FP32 accumulation using FMA product
residuals and full TwoSum, following [Dot2](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).
Exact or missing tolerances retain ordinary arithmetic. Eligibility excludes
observed carriers and coupled online-normalization state. This policy resolved an
observed cancellation failure; it changes rounding and can slow attention.
The current implementation retains deferred Dot2 finite guards until a carrier
is read, preserving the same arithmetic and nonfinite-state behavior. In the
retained phase 3 study, across 144 passing
paired runner responses, this reduced measured large-matmul latency by **7.38%**
and the first attention kernel by **8.70%**. Results are indicative measurements
on the runner, not guarantees for every shape. A scalar-load-packing experiment
was rejected after it showed no broad runtime gain. The
[performance study](docs/PERFORMANCE.md) records the proof, measurements and
tradeoffs. Scheduled unrolling has per-loop and per-kernel size budgets.

Phase 8 adds `.maxntid` from the supplied block dimensions, including helper
threads. A paired experiment against our frozen previous compiler found a
2.60x median improvement on `a02_attention_128`, while the other sampled public
workloads were approximately unchanged. The [phase-8 measurements](docs/PHASE8.md#performance-experiment)
retain rounds, input checks and limits; this is not a ranking against other
submissions or a claim for every shape. The exact remainder fallback may be
slow for extreme exponent ratios. Existing approximate exp and sigmoid/SiLU
formula policies remain unchanged and explicit in the [type rules](docs/TYPE_SUPPORT.md).

A separate [selector pilot](docs/SELECTOR_STUDY.md) explores fixed heuristics,
a shallow tree and offline bandit replay over existing unroll budgets. Its
[independent review](docs/SELECTOR_REVIEW.md) records the correctness, provenance
and held-out selection gates. The sealed 30-workload pilot was rejected before
training or comparative GPU timing: only 1/10 training workloads, 0/10 validation
workloads and 1/10 final workloads produce different PTX across the budgets.
The [static audit](artifacts/validation/phase4/selector/feasibility.json) also
confirms all 90 frozen-versus-production PTX comparisons match. These are CPU
emission comparisons, not GPU correctness checks. The compiler retains fixed
budget 1024; no learned selector is promoted and no selector speedup is claimed.

## Reproduce validation and retained phase-4 evidence

```console
python tools/check_compiler.py --mode remote --seeds 0,17 --report build/remote.json
python tools/check_release.py --compiler compile.py --report-dir build/release-recheck
python tools/check_phase4.py --compiler compile.py --report-dir build/phase4-recheck
python tools/check_clean_install.py
```

The complete runner gate requires the retained generated packages and a new,
unused report directory. Clean-environment setup uses
`python tools/check_clean_install.py --install` on its first run.

The retained 23 October phase-4 gates use backend
`5cb3a351840ca03055b9725471fd3b44afee7a1d8e2832c49d439018da5ba9fc`.
Its [complete official-runner gate](artifacts/validation/phase4/release-remote/summary.json)
passes **213/213 numerical checks across 97 workload packages**. The
[new semantic/interaction gate](artifacts/validation/phase4/semantic-final/summary.json)
passes **29/29 checks**: 23 additional packages under the default compiler,
four rolled interaction variants and two synchronous-copy variants. Together
these are **236 default checks across 120 packages, plus six variant checks**,
or **242 passing checks**. Variant executions are not extra distinct programs.

Separately, the sealed selector corpus passes
[90/90 additional correctness-only checks](artifacts/validation/phase4/selector-correctness-retry01/summary.json)
across 30 new workload packages with the same production compiler. Its
[offline integrity audit](artifacts/validation/phase4/selector-correctness-retry01/integrity-audit.json)
revalidates every returned output and all 120 request/response events. Combined
coverage in that 23 October snapshot is **332 passing checks across 150 packages: 326 default checks plus
six variants**. The archive's original five reports and 242-check gate remain
unchanged; these 90 checks are additional evidence, not a selector speed study.
An earlier report-write failure is retained and excluded from these totals.
The [final evidence index](artifacts/validation/phase4/final-summary.json)
links the frozen release gates and the additional audited run separately.

The CPU/assembler suite records 73 methods: **55 passed and 18 local-GPU methods
intentionally skipped**, with no failures or errors. Clean installation passes
21 deterministic CLI invocations plus three diagnostic checks. All GPU work in
this phase used the official runner. Historical sanitizer results remain
source-labelled evidence; there is no new phase 4 sanitizer claim.

The earlier phase 3 local 46-method run retained one pre-launch harness failure for a
dynamic shared-memory case: 183 assembly checks and 168 actual GPU executions
passed, but the entire local run did **not** pass. The harness incorrectly
applied a 48-KiB cap to a device supporting 64 KiB with opt-in. That same kernel
passes on the official runner. Subsequent GPU validation uses that runner;
no passing local rerun is claimed. The phase 3 API rejections and first failed
transport bridge also remain visible.
The older **157/157** gate belongs to backend `0373bbc6...`, not this source version.
These are separate measurements, not additive unique-program counts.

Read [VALIDATION.md](docs/VALIDATION.md) for complete commands, source hashes, reports,
target-adaptation limits, and current status; [SEMANTICS.md](docs/SEMANTICS.md) for
the node contracts; and [COMPILER.md](docs/COMPILER.md) for architecture and limitations.
Passing these checks does not establish hidden-set correctness or a winning score.
