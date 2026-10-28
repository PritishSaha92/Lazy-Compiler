# Validation

The compiler is checked through the required `python compile.py kernel.ir.json`
interface. Assembly success, numerical correctness, and timing are separate
results. Tests never replace the submitted translation with Emmy's renderer,
nvcc, LLVM, Triton, or a reference CUDA file.

## Current phase-9 revision

See [PHASE9.md](PHASE9.md) and the [release summary](../artifacts/validation/phase9/release-summary.json).
The current source passed **15/15 public and 67/67 typed feature packages**,
82 distinct packages in total. CPU/assembler discovery passed **129 of 148
methods**, with 19 explicit local-GPU skips and unchanged source fingerprints.
The feature report's offline audit verified 67 actual CLI outputs and 134 HTTP
exchanges. The public run initially completed 14 cases; a connection reset
prevented output on row softmax, and its unchanged-source retry passed.

The first new-feature run executed and passed eight GELU and six fragment
packages. Two additional requests were rejected with HTTP 422 before execution
due to nested input lists. Original data and requests remain preserved. Flattening
transport lists did not alter values, IR, expected arrays or tolerances; the
fresh 67-package gate then executed both successfully.

The baseline and final eight-package generated-composition experiments each
passed **21/24 mathematical checks**. The same three FP16 erf-GELU checks failed
the original tolerance; all 5,439 affected final-run values match an independent
[typed-stage half oracle](../artifacts/validation/phase9/numerics/half-gelu-stage-final.json)
exactly. These are retained precision-contract conflicts, not passed
mathematical comparisons. Final source fingerprints stayed unchanged. Direct-op
tests and generated compositions are distinct: Emmy decomposes the latter into
simpler arithmetic. The release summary explicitly records that not all numerical
references pass.

The [final composition integrity audit](../artifacts/validation/phase9/compositions-final/integrity-audit.json)
replays the retained evidence; it does not turn the three mathematical failures
into successes or add new GPU executions.

No new timing advantage is claimed. The phase-8 launch-bound measurements and
earlier validation results below remain historical evidence with their original
source hashes.

## Historical phase-8 revision

The [phase-8 report](PHASE8.md) covers prescribed launch bounds, signed integer
loop bindings, floating floor remainder and the exact integer output oracle.
A fresh run through the unchanged organizer `submit.py` passed **15/15 public
problems**. The final additional runner gate passed **51/51 packages**:
41 retained, six loop-binding and four remainder. CPU/assembler discovery passed
**120 methods**, with 19 local-GPU skips out of 139 discovered methods and no
failures or errors. The [phase-8 release summary](../artifacts/validation/phase8/release-summary.json)
records source hashes, these completed gates, raw-response replay and checkout
verification. No local GPU was used for this revision.

The initial broad gate passed **49/51**. Its two FP64 raw-I64 fixture failures
remain recorded: the runner's binary64 numeric conversion lost low bits.
The final roster replaces those two fixtures with two-I32-word transport;
references and zero tolerance are unchanged. This is a test-transport repair,
not a conversion of the original failed attempts into passes. See
[PHASE8.md](PHASE8.md) for both runs and their attribution.

The four remainder packages use 528 input pairs per type through both named
aliases. FP32 uses one I32 word and FP64 uses two I32 words per value to preserve
raw IEEE bits through the runner's numeric conversion. The output
oracle compares zero-tolerance integers without floating-point coercion,
including mixed signed/unsigned 64-bit values. NaN outputs are explicitly
canonicalized for these fixtures rather than accepted by the ordinary floating
comparison policy. [Oracle tests](../tests/test_integer_oracle.py) cover the
checker independently of the backend.

A separate exact rational/IEEE-encoding audit passed for all **1,056 FP64
remainder values** in the two word-transport packages. It checked **4,224 input
words and 2,112 output words** through the observed binary64-to-I32 wire model.
That is independent reference/transport evidence, not another GPU execution.

The phase-8 paired launch-bound experiment is performance evidence against our
previous compiler, separate from the correctness gate. See its retained
[timing reports and limitations](PHASE8.md#performance-experiment).
Earlier phase reports keep their original sources, results and failures below.

## Historical phase-7 release

The [phase-7 release](PHASE7.md) corrects logical builtin names/types,
unstamped mixed-integer operations and expression-shift promotion. Fresh checks
passed **15/15 public + 41/41 additional organizer-GPU packages**: 29 previous
feature examples, 11 new counterexamples and one retained scalar case whose PTX
changed. Complete discovery passed **110 CPU/assembler methods**, with 19 explicit
local-GPU skips. The isolated install passed 25 packages, 75 compilations and
three expected diagnostics. The [source-bound
summary](../artifacts/validation/phase7/release-summary.json) links all reports.
No local GPU was used. These results belong to the phase-7 sources, not the
changed phase-8 backend. The following phase-6 results likewise belong to their
previously packaged sources and are retained unchanged.

## Previously packaged phase-6 research-informed release

See [PHASE6.md](PHASE6.md): **15/15 public** and **29/29 optional-feature**
organizer-GPU checks; **99 CPU/assembler methods passed**, with 18 explicit
local-GPU skips; and 14 isolated-install packages. The [research decision
record](RESEARCH_REVIEW.md) explains the architecture guard and new ring/isolation
regressions. The [phase-5 report](PHASE5.md) is retained as earlier evidence.

## Previous scalar-only runner review (24 October 2026)

That review used production source `5cb3a351840ca03055b9725471fd3b44afee7a1d8e2832c49d439018da5ba9fc`.
The [dated review](FINAL_CHECK_2026_10_24.md) and
[closed evidence index](../artifacts/validation/2026-10-24-final/summary.json)
record **79 executed checks: 70 mathematical-reference passes and 9 retained
mathematical-reference failures**. This is not an all-passing extra-corpus gate.

| Scope | Executed | Mathematical pass | Mathematical fail |
|---|---:|---:|---:|
| [Unmodified public `submit.py` client](../artifacts/validation/2026-10-24-public-submit-runner/summary.json) | 15 | 15 | 0 |
| Original generated graphs: 16 graphs, three inputs each, plus one explicit large-I32 input variant | 49 | 40 | 9 |
| Two separately hand-built ceil/remainder operation fixtures, three inputs each | 6 | 6 | 0 |
| Three separately constructed FP64-state IR variants, three inputs each | 9 | 9 | 0 |
| Total this review | 79 | 70 | 9 |

The 64 checks outside the public set therefore have **55 mathematical passes and
9 mathematical failures**. The generated cases cover all five categories, real
FP16/FP64/I32 buffers, irregular dimensions and operations absent from the public
set. Generator failures, ABI drift and an out-of-scope tensor-core schedule are
retained separately; cases that never reached execution are not counted as
backend GPU failures or passes.

The nine executed failures are FP64-buffer sum, matmul and centered-normalization
cases with **FP32 state/arithmetic in the generated scheduled IR**. The
[precision attribution](../artifacts/validation/2026-10-24-extra/precision-attribution/summary.json)
rechecks the existing raw outputs against explicit FP32 IR oracles: all nine
match bit-for-bit. Across their 3,057 output elements, the closest FP32 value
already violates the original mathematical tolerance for 3,052 elements. This
is evidence of an incompatible precision contract, not a reason to silently
widen the submitted translation or weaken the reference tolerance.

The [explicit FP64-state variants](../artifacts/validation/2026-10-24-root-audit/fp64-state-variants/manifest.json)
are clearly labelled **test-derived IR**, not unchanged generator outputs.
They pass 9/9 checks with the original expected arrays and original tolerances.
They establish FP64-state lowering on those fixtures; they do not convert the
nine original failures into passes. The two hand-built operation fixtures are
also distinct from successful `tools/generate.py` outputs.

The final review's top-level `pass` means the stated release review criterion
was met: compatible-reference tests passed and all nine retained failures were
attributed to the actual IR precision. Its `original_extra_math_gate_pass` remains
`false`. The index SHA-256 is
`8501ccd7f89d1aa34dbe64433289195379945489bb5d72bb9f693401b48e3e06`.
No production source changed. The
[phase-4 manifest](../artifacts/release/phase4/submission-bcbc4a8381497f1d34be-3908fd80497591da4d12.manifest.json)
adds this review to six retained evidence snapshots while referencing the same
three-file ZIP. The public and extra checks used only the organizer's GPU.

The 23 October 332-check snapshot below is retained separately. Repeated public
checks are not appended to it as new unique coverage, and no combined all-pass
claim is made. These remain development workloads, not the unavailable hidden
set or a comparative performance result.

## Historical frozen-candidate gate (23 October 2026, phase 4)

The source hashes for this historical snapshot, unchanged on 24 October, are:

| File | SHA-256 |
|---|---|
| `ptx_backend.py` | `5cb3a351840ca03055b9725471fd3b44afee7a1d8e2832c49d439018da5ba9fc` |
| `compile.py` | `4ed3fb67833b7de20693c7cb3f163a4646cac8d3a71e63649c1af1bccd37ce56` |
| `requirements.txt` | `5d054f8757b1212865653105a4e7c634e2952007da020d2f67ad456eb6bccc84` |

This historical run passed **326 default numerical checks across 150 distinct workload
packages**, plus **six lowering-variant checks**: **332 passing checks in total**.
All GPU execution in this phase used the official development runner. Repeated
seeds and variants are additional checks, not additional distinct programs.

| Final source-matched gate | Default checks | Variant checks | Distinct packages | Result |
|---|---:|---:|---:|---|
| [Complete regression inventory](../artifacts/validation/phase4/release-remote/summary.json) | 213 | 0 | 97 | 213/213 PASS |
| [Type and interaction additions](../artifacts/validation/phase4/semantic-final/summary.json) | 23 | 6 | 23 | 29/29 PASS |
| [Additional sealed-corpus correctness](../artifacts/validation/phase4/selector-correctness-retry01/summary.json) | 90 | 0 | 30 | 90/90 PASS |
| Combined | 326 | 6 | 150 | 332/332 PASS |

The first two rows are the **242 checks across 120 packages** bound to the
immutable release archive's five reports. The third row is separately retained
evidence for the same source; it was not retroactively embedded in that archive.
The [final evidence index](../artifacts/validation/phase4/final-summary.json)
records this accounting and the excluded incomplete attempt.

The first row reruns the complete phase 3 inventory against this source. It
includes 75 public checks, 20 earlier generated checks, eight layout fixtures,
54 earlier corpus checks, 15 scheduled workload checks, nine native
transport/shared-memory fixtures, two product reductions and 30 typed semantic
fixtures. The four unsigned-buffer cases use the previously verified signed
32-bit word transport, integrated before execution. The
[transport manifest](../artifacts/validation/phase4/release-remote/input-transport.json)
retains byte, IR, expected-output and PTX identity. This new aggregate is fully
passing; the older API rejections and failed bridge attempts below remain failed.

The 23 additional packages comprise 19 type fixtures and four interaction
fixtures. They exercise narrow expression rounding, BF16 conversion/comparison,
nonfinite predicates, small integer selection and bitcasts, unsigned unary
operations and signed magnitude population count. Interaction cases cover
guarded invalid addresses with live state, ragged row alignment and scalar
cleanup, and a two-slot, seven-stage asynchronous pipeline with cross-warp
consumers and a final drain. The six variants are four rolled interaction
kernels and the asynchronous fixture with synchronous copies in both rolled
and unrolled forms. The default asynchronous case still uses native `cp.async`.
Source and executed input/helper hashes remained unchanged in both final gates.

The additional sealed-corpus run checks three explicit input cases for each of
30 packages, using the production default with zero warmups and one iteration.
Its [offline integrity audit](../artifacts/validation/phase4/selector-correctness-retry01/integrity-audit.json)
revalidates all 90 outputs against the frozen independent references, all 120
compile/run request-response events and their HTTP-attempt artifacts. There
are no pending or unreferenced events. Source and seal hashes match, and the
completed run recorded no transport or report-writer retries. This is numerical
correctness evidence; it does not train a policy or establish a runtime gain.

The first attempt stopped on `PermissionError: [WinError 5]` while atomically
replacing its summary. Its
[retained partial audit](../artifacts/validation/phase4/selector-correctness/integrity-audit.json)
checks 22 recorded passing outputs and separately validates one already-written
response that had not reached the aggregate. The attempt remains incomplete:
neither the 22 recorded checks nor the recovered extra response contributes to
the fresh 90-check total. No kernel mismatch was reported and no particular
reader or file-lock cause was proved. The new driver adds bounded summary-write
retries without retrying GPU requests. Its
[seven passing mocked tests](../artifacts/validation/phase4/selector-preparation/correctness-driver-final-tests.txt)
cover transient and persistent write failures, output/entry/source failures,
and the offline auditor's rejection of a corrupted raw response. The original
driver, raw artifacts and `.tmp` summary remain beside the incomplete report.

The [CPU/assembler report](../artifacts/validation/phase4/cpu-regressions.json)
and [transcript](../artifacts/validation/phase4/cpu-regressions.log) record
**73 methods: 55 passed, 18 intentional local-GPU skips, zero failures and zero
errors**. Skipped methods require the opt-in `EMMY_GPU_TESTS=1`; they are not
counted as passing executions. This phase requested no local GPU execution.
The [clean-install report](../artifacts/validation/phase4/clean-install/summary.json)
passes **21 deterministic CLI invocations plus three diagnostic checks**, with
only the three submission files staged and source hashes unchanged. Guarded
invocations observed no prohibited renderer calls, subprocesses, network use
or source-repository reads. These are same-host packaging and emission checks,
not additional GPU results or a hash-locked dependency guarantee.

The [phase-4 public PTX export](../build/open-ptx-phase4-cli/manifest.json) uses
actual CLI stdout bytes and separate problem directories. The
[identity report](../artifacts/validation/phase4/public-ptx-identity.json) verifies
all **15/15** hashes against official-runner payloads. The earlier
`build/open-ptx-phase4` export had colliding filenames and newline differences;
it is superseded and must not be cited as the current exact-byte export.
The [phase 4 archive manifest](../artifacts/release/phase4/submission-bcbc4a8381497f1d34be-b88459958203b4fdede2.manifest.json)
binds the three source files and five final evidence reports to the release.

The [type contract](TYPE_SUPPORT.md) and
[transformation contracts](TRANSFORMATION_CONTRACTS.md) state the limits of the
new admission checks and lowering paths. Recognized types do not imply support
for every operation. Narrow composite arithmetic uses the documented FP32
expansion/narrowing policy; this is not a correctly rounded general math library.
The structural analyzer is development infrastructure, not a bounds or
coalescing proof. No new phase 4 sanitizer run is claimed. Historical sanitizer
and performance studies below remain tied to their original sources.

These are curated development regressions, not the organizer's hidden set.
Previously frozen corpora are now regression inputs. The separate
[selector pilot](SELECTOR_STUDY.md) was rejected by its static opportunity gate:
only 1/10 training, 0/10 validation and 1/10 final workloads have nonidentical PTX
choices. Its [90/90 production-emission comparisons](../artifacts/validation/phase4/selector/feasibility.json)
are CPU checks, not GPU numerical executions. No learner was trained, no
comparative timing experiment was run and no learned policy was promoted.
The existing budget 1024 remains in production. See its
[review](SELECTOR_REVIEW.md) for the evidence and selection protocol.

## Historical frozen-candidate gate (23 October 2026, phase 3)

Everything in this section belongs to the phase 3 source below. It is retained
to explain the transport fixes and prior failures, not added to phase 4 totals.

Backend SHA-256:
`8afa55d5ac32da2eda6184058de9aa40ee8327366fc9fa6261b82fb0ed2ce002`.
At phase 3 completion the planned official-runner inventory was covered by **213 successful
numerical checks across 97 logical workload packages**: 209 original checks plus
four verified integer-word transport checks. The
[composite audit](../artifacts/validation/phase3/release-summary.json)
links the separate reports and preserves the failed attempts. The first expanded
gate completed with **209 passing numerical executions and four pre-execution
API rejections** out of 213 requested cases. Its
[original failed aggregate](../artifacts/validation/phase3/release-remote/summary.json)
is retained with the frozen compiler, input/reference/helper hashes, serialized
jobs and per-job reports. It is not an all-passing 213-case gate.

The four alignment-overflow fixtures use `u32` or `u64` input descriptors, which
the runner API rejects; it accepts `f16`, `f32`, `f64`, `i32` and `i64`. No kernel
was executed for those four requests. The first separate transport bridge
represents the same input bytes using same-width signed descriptors, while
retaining the IR, PTX, shapes and independent expected outputs. Its
[byte-identity manifest](../artifacts/validation/phase3/unsigned-transport/adaptation-manifest.json)
proves client-side identity, and its
[execution report](../artifacts/validation/phase3/unsigned-transport/remote.json)
records two passing `u32` cases and two `u64` device errors. This failed bridge
attempt remains retained and does not contribute extra cases to the final total.

A bounded-address bit-roundtrip diagnostic then isolated the transport issue.
The [same-width signed wire](../artifacts/validation/phase3/word-transport-diagnostic/old-remote.json)
completed safely but returned two incorrect words out of sixteen, with maximum
absolute error 343. The
[32-bit word wire](../artifacts/validation/phase3/word-transport-diagnostic/new-remote.json)
returned all sixteen words exactly using the same PTX and cubin. The observed
lost low bits are consistent with a floating-point conversion of large JSON
integers; the runner's private implementation was not inspected, so that
internal mechanism remains an inference.

The final bridge transports integer input buffers as signed 32-bit words. For
64-bit inputs the wire shape has twice as many elements; the allocated byte
count, device input bytes, IR types and kernel argument layout remain identical.
Its [manifest](../artifacts/validation/phase3/unsigned-word-transport/adaptation-manifest.json)
records exact identity of original/emitted PTX, independent expected arrays and
every input byte. All **four original alignment kernels pass** through this
bridge on the official GPU in the
[final transport report](../artifacts/validation/phase3/unsigned-word-transport/remote.json).
The production compiler did not change. The original aggregate remains failed;
213 is the combined successful coverage, not a claim that its initial attempt
passed.

| Official gate | Distinct packages | Requested cases | First-attempt result |
|---|---:|---:|---|
| Public problems, two normal seeds and three stress patterns | 15 | 75 | 75 PASS |
| Earlier generated workloads, two seeds | 10 | 20 | 20 PASS |
| Existing layout fixtures | 8 | 8 | 8 PASS |
| Earlier frozen corpus, two normal seeds and one scaled seed | 18 | 54 | 54 PASS |
| New scheduled workloads, two normal seeds and one scaled seed | 5 | 15 | 15 PASS |
| Native transport/shared-memory fixtures | 9 | 9 | 9 PASS |
| Nonempty product reductions | 2 | 2 | 2 PASS |
| Typed semantic and memory regressions | 30 | 30 | 26 PASS; 4 API rejections |
| Total | 97 | 213 | 209 PASS; 4 API rejections |

The separate final word-transport run completes the four rejected requests,
giving 213 successful checks over the same 97 logical packages. Adapted package
copies and diagnostic executions are not counted as additional logical cases.

These are curated development regressions, not organizer-hidden tests. The
earlier frozen corpus is now reused for regression and performance checks, so
the current iteration does not claim it as untouched evaluation. The final gate
uses the production CLI and native runner target, including asynchronous copies.
Its one-warmup/three-iteration timings are correctness feedback, separate from
the paired performance experiments.

The continued audit fixed four generic issues, each with retained counterexamples:

- `StridedLoop.end` must be captured at entry when it references a mutable
  carrier; the loop must not silently reevaluate the changed carrier.
- Legacy rank-mismatched buffer indices use the installed IR's sum-of-offsets
  convention, with sufficiently wide arithmetic before summing large offsets.
- Arithmetic on predicate operands must promote to an integer type, while
  logical and selection operations retain predicate semantics.
- Integer alignment facts must respect fixed-width wraparound. After wrapping
  addition, subtraction or multiplication, only common divisors with the
  integer modulus remain proved; otherwise later division can produce a
  misaligned index falsely advertised as aligned.

The selected Dot2 finite-guard deferral preserves product/residual/TwoSum order
and all observable state. Its two paired runner studies pass **144 responses**
across eight workloads, including measured latency reductions of 7.38% for the
large public matmul and 8.70% for the first public attention kernel. All returned
timed outputs were checked; each response contains repeated launches rather
than an output check after every individual launch. Automatic scalar-load
packing was rejected after 108 passing comparison responses showed no broad
runtime improvement. See [PERFORMANCE.md](PERFORMANCE.md) for complete protocols,
source hashes, proofs, numerical-policy costs and indicative timing limits.

### Local result retained as a failure

The [46-method local run](../artifacts/validation/phase3/release-local.json) ended
with **one failed subcase and no skipped methods**; its
[complete log](../artifacts/validation/phase3/release-local.log) is retained.
Every assembly method passed. Combining that transcript with the independent
[coverage accounting](../artifacts/validation/phase3/local-coverage.json) gives
183 successful assembly checks and 168 successful numerical GPU executions.
The planned inventory contains 132 typed programs and 169 numerical executions;
one of those planned executions never launched. These counts do not turn the
overall failing unittest result into a pass.

The failure was `dynamic_pool_distinct_alignment`, which requests 49,284 bytes
of dynamic shared memory. The local harness rejected it before loading or
launching its kernel because it hardcoded a 48-KiB limit for the older GPU. A
[read-only capability query](../artifacts/validation/phase3/local-shared-capability.json)
reported a 65,536-byte opt-in per-block limit on the GTX 1650 Ti. This identifies
a harness limitation, not a numerical mismatch. The same emitted kernel now
passes independently on the official GPU in the
[nine-case native transport gate](../artifacts/validation/phase3/release-remote/transport.json).
That verifies the compiler output on the runner; it does not certify the
corrected local harness path. Compiler and executed fixture hashes were unchanged
during the recorded local run. Following the user's preference, subsequent GPU
validation uses the official runner; no successful local rerun is claimed.

The local capacity helper was subsequently corrected to query the device's
default and opt-in limits. Its
[promotion record](../artifacts/validation/phase3/local-capacity-candidate/promotion.json)
records five passing CPU mock methods covering eleven scenarios, an archived
original helper, unchanged remote helper functions and unchanged compiler files.
There were zero GPU executions for this correction. These mock checks do not
certify the corrected actual-driver path or replace the failed local run.

### Historical phase 3 clean-package and sanitizer checks

The phase 3 backend separately passed
[21 deterministic clean-environment CLI compilations and three diagnostic checks](../artifacts/validation/phase3/clean-install/summary.json).
Only the three submission files were staged; seven anonymized inputs cover all
five families plus generated matmul/attention. Guarded invocations observed no
renderer calls, backend subprocesses, prohibited network access or reads from
the original repository. This is packaging/emission evidence, not extra GPU
execution. The same-host environment uses pinned Emmy 0.3.11; other dependencies
remain version-recorded rather than fully hash-locked.

The [phase 3 memory sanitizer report](../artifacts/validation/phase3/memory-sanitizer/summary.json)
passes **15 instrumented numerical executions**: five typed memory cases under
each of `memcheck`, `racecheck` and `synccheck`, with zero reported errors,
hazards or warnings. The cases exercise aligned/shifted vector stores, an
intervening input-buffer write, a conditional write, and reused value bindings.
The suite retains its historical `load-packing` name, but it ran the phase 3
production compiler, which did not enable the rejected packing transformation.
These checks used explicit target adaptation on the older local GPU and contain
no asynchronous copies; they do not certify native `cp.async` or all schedules.
They completed before the subsequent official-GPU-only preference. They are
separate instrumented invocations, not 15 additional distinct programs.

## Historical frozen-candidate gate (23 October, phase 2)

The results in this section belong to backend SHA-256:
`0373bbc64340ae37274a67a92deb7320915ff4aa5cef97a231b3ac1ec7a2cc59`.
That default compiler passed **157/157 official-runner numerical cases**: the
[103 public/stress/generated/layout checks](../artifacts/validation/phase2/release-original-gates.json)
and [54 additional held-out checks](../artifacts/validation/phase2/release-holdout-default/summary.json)
across 18 new workloads. The second corpus and candidate were frozen before
evaluation; its manifest and source hashes were verified. These are development
holdouts, not the organizer's undisclosed hidden set. Any strict-policy ablation
is a separate numerical-policy experiment and is excluded from these totals.

Its local suite passed **28 unittest methods with no skips**, covering
84 distinct typed IR programs, 116 assembly checks, and 111 numerical GPU
executions across the selected modes. The
[scoped provenance review](../artifacts/validation/phase2/release-local-reviewed.json)
confirms that the compiler, executed test fixtures, and their helpers remained
unchanged. The original wrapper flagged a concurrent edit to the unrelated,
nonexecuted `tests/evaluate_holdout.py`; its failure report is retained. The
review validates the already successful unchanged-code test results rather
than concealing that broad directory-fingerprint failure.

Separate checks below report 21 clean-environment compilations plus three CLI
diagnostics, and nine named-barrier sanitizer executions. Those are additional
invocations of existing inputs, not extra unique programs. The dated 22 October
section retains historical results and source hashes; its counts are not the
current source's totals.

## Reproduce the checks

Start with the current [phase-9 status](PHASE9.md) and source-checkout verification:

```console
python tools/check_checkout.py --report artifacts/validation/phase9/checkout/my-verification.json
python tools/run_regressions.py --report artifacts/validation/phase9/checkout/my-cpu-regressions.json
```

The release's [checkout report](../artifacts/validation/phase9/checkout/verification.json)
uses the name `verification.json`. The commands above use fresh reviewer names;
use another `--report` path for repeated checks. These commands do not contact
the organizers' runner.

The broader historical corpus can be checked separately with the following
commands. They use retained generated packages, require unused report
directories and serialize GPU launches on the official runner:

```console
python tools/check_release.py --compiler compile.py --report-dir artifacts/validation/release-recheck
python tools/check_phase4.py --compiler compile.py --report-dir artifacts/validation/phase4-recheck
```

The first command checks the 213-case regression inventory; the second builds
the 23 type/interaction packages and checks their 23 default plus six variant
executions. Both require unused report directories and use the official runner.

The wrapper now preserves the original `typed-inputs/`, creates the four
unsigned-buffer wire packages under `typed-wire-inputs/`, and records byte,
IR, expected-output and PTX equality in `input-transport.json`. No additional
flag is required. This integrates the verified transport bridge into future
runs; it does not rewrite the earlier failed report. The
[wrapper CPU check](../artifacts/validation/phase3/unsigned-word-transport/wrapper-cpu-proof.json)
passes seven tests, including eight real CLI compilations and a mocked complete
213-case/97-package workflow. No GPU calls were made for that wrapper CPU check.
The new phase 4 aggregate separately records 213/213 actual numerical checks;
the historical phase 3 combined coverage remains in its separate reports.

The following commands select smaller gates. Unit discovery without
`EMMY_GPU_TESTS=1` runs CPU and assembly checks and reports GPU methods as skipped;
it is not a substitute for GPU validation.

```console
python -m unittest discover -s tests -v
python tools/check_compiler.py --mode assemble
python tools/check_compiler.py --mode remote --seeds 0,17 --warmup 1 --iters 3 --report artifacts/validation/final-public.json
python tools/check_compiler.py --mode remote --seeds 101 --patterns ones,alternating,wide --warmup 1 --iters 3 --report artifacts/validation/final-public-stress.json
```

To check the ten generated workload packages after running the generator below:

```powershell
$freshProblems=(Get-ChildItem -LiteralPath 'artifacts/validation/stress' -Directory).FullName
python tools/check_compiler.py --mode remote --seeds 0,17 --references tests/expressions_stress.py --warmup 1 --iters 3 --report artifacts/validation/final-fresh.json @freshProblems
```

The exported independent layout fixtures can also be run with native
asynchronous copies on the official GPU:

```powershell
$layoutProblems=(Get-ChildItem -LiteralPath 'artifacts/validation/layouts' -Directory).FullName
python tools/check_compiler.py --mode remote --seeds 0 --warmup 1 --iters 3 --report artifacts/validation/final-layouts.json @layoutProblems
```

The remote mode calls the organizer's `/compile` and `/run` development APIs. It
does not submit an entry, publish a repository, or contact the organizers. Calls
are serialized. Each atomic output is reset before every launch, including
warmups, according to `zero_outputs`. Input pointers follow `arg_order` even if
the metadata's argument records have a different order. Intermediate buffers are
carried between kernels in metadata order.
Connection failures and transient server errors receive bounded retries; failed
infrastructure attempts remain distinct from numerical mismatches. Every report
records the emitted PTX hash and the compiler source hashes. Compilation fails
if either compiler source changes during its invocation.

The checker compares every returned element with the independent NumPy
references and the package's `rtol`/`atol`. Output shape mismatches fail instead of
using NumPy broadcasting. Missing/truncated outputs fail. Matching NaNs do not
count as a pass. Zero-tolerance integer references require integer outputs and
use exact value comparisons, avoiding float64 rounding above `2**53`. Integer
first-failure values remain integers in JSON. Seed zero uses the supplied `expected.npz`; additional seeds use
the mathematical references in `tools/expressions_open.py`. Additional spec files
can be supplied using `--references`.

## Local CUDA execution

Local execution is optional infrastructure retained for other environments.
The current continuation uses the official runner. The failed local shared-memory
capacity check above remains visible; no local rerun is implied by these recipes.

```powershell
$env:EMMY_GPU_TESTS='1'
python -m unittest discover -s tests -v
python tools/check_compiler.py --mode local --adapt-local-target
```

The local driver harness uses `ctypes` and a separate worker process for each
kernel. It initializes ordinary floating outputs to NaN, which exposes missing
writes that a zero-filled output could hide. It executes the emitted PTX through
the CUDA driver, without compiling CUDA C++.

The development machine has a GTX 1650 Ti (`sm_75`), CUDA 12.6 `ptxas`, and Emmy
0.3.11. The contest's GPU is an RTX 5090 (`sm_120`), while problem schedules target
`sm_89`. `--adapt-local-target` changes only the target directive for instructions
the older GPU supports. It explicitly skips PTX containing `cp.async`, MMA, or
mbarrier operations. A skipped case is not reported as passing. The submitted
compiler's default PTX and the remote checks retain their intended target and
native asynchronous copies. Local timings include launch and synchronization
overhead and are not kernel-only benchmarks.

## Counterexamples and genuinely new schedules

`tests/ir_cases.py` constructs eleven small typed-IR programs with independent
references. They exercise odd tile extents, conditional branches, ternary
indices, output-first reordered parameters, scalar/v2/v4 loads and stores,
`Accum.base`, `Loop.seed=False`, nested accumulator lifetimes, subgroup shuffles,
a partial final warp, and segmented shared-memory reduction broadcasts.
`tests/test_semantics.py` adds further cases for fields absent from the public
set. These tests use typed Emmy objects, not textual editing of IR body strings.

```console
python tests/ir_cases.py --out artifacts/validation/synthetic
python tests/generate_stress.py --out artifacts/validation/stress --target sm_89
```

`tests/expressions_stress.py` defines ten additional workloads with different
shapes and operation combinations, including prime-sized pointwise/reductions,
row tails, maximum reduction, nonlinear pointwise operations, normalization,
softmax, rectangular/tail matmuls, and rectangular attention. The generator is
development infrastructure: Emmy creates the scheduled input IR and matching
launch metadata; our compiler performs the IR-to-PTX translation under test.

On Windows, Emmy's CLI imports Unix `fcntl` from its optional golden-inventory
module. `tests/generate_stress.py` provides an isolated import-only shim with
golden inventories disabled. Any attempted lock raises an error; no lock is
silently bypassed, and no installed package is modified.

## Recorded results (22 October 2026)

The release workload PTX passed **103 of 103 numerical cases** on the official
RTX 5090 runner. Each case used one warmup and three measured launches, including
split-K output resets. All five problem categories are represented. The original
three execution gates contain 95 cases; a later native-layout gate adds eight.

| Gate | Workloads | Numerical cases | Result | Report under `artifacts/validation/` |
|---|---:|---:|---|---|
| Public problems, seeds 0 and 17 | 15 | 30 | PASS | `final-public.json` |
| Public problems, ones/alternating/scaled normal inputs at seed 101 | 15 | 45 | PASS | `final-public-stress.json` |
| Fresh generated workloads, seeds 0 and 17 | 10 | 20 | PASS | `final-fresh.json` |
| Independent layouts, native asynchronous copies and warp synchronization | 8 | 8 | PASS | `final-layouts.json` |

The original three execution gates and every per-kernel compilation in them
recorded these same SHA-256 hashes:

```text
compile.py      4ed3fb67833b7de20693c7cb3f163a4646cac8d3a71e63649c1af1bccd37ce56
ptx_backend.py  8fbe34ef6ce9df95bca1181838ea2229dee8485cae016247117551cd7db5d91e
```

A subsequent warp-barrier fix changed the backend hash to
`3e68e17421a6f620eb95f29395989eda26f695683d353c4f2b3c28194a46c36b`.
The 25 public/fresh IRs contain no warp-only `Sync` operation. Recompiling every
one through that revision's CLI produced **25 of 25 byte-identical PTX outputs**
against the execution reports; the stress report used those same public PTX
outputs. `final-translation-identity.json` records this provenance, the original
report hashes, and the recorded IR/metadata hashes. The check used
`tests/verify_recorded_ptx.py`; it is specific to those historical revisions,
and later intentional code-generation changes have their own execution gates.
This preserves the recorded GPU evidence
for those workloads; it does not claim that a changed path was executed by the
earlier gates. The new warp-barrier path has a separate targeted test below.
The later layout gate executed native PTX from that backend revision directly on
the RTX 5090: all eight cases passed exact comparisons, including all four
asynchronous-copy swizzles and the corrected warp barrier. It adds independent
evidence for paths absent from the original 95 cases; it is not part of the
25-output translation-identity comparison.

`final-summary.json` records the counts and report hashes. The first public
attempt encountered two connection timeouts, for the two attention workloads.
Only those affected workloads were retried, with the same compiler hashes;
both passed. The consolidated report links `final-public-attempt1.json` and
`final-public-retry.json`, preserving the original infrastructure failures.

That 22 October source's local suite completed **20 unittest methods with no
skips in 170.362 seconds**, including 67 `sm_89` assembly checks and 62 numerical
GPU executions over 45 distinct typed-IR programs. All 20 methods ran together
after the warp-barrier correction. `final-unittest-current.log` is the transcript;
`final-unittest-current.json` records unchanged source hashes before and
after the run. The earlier 17-method transcript remains in `final-unittest.log`.
The included `tests/test_policy.py` method makes three actual CLI/GPU executions
using an existing cancellation fixture. Missing and zero tolerances select
strict FP32 behavior; nonzero tolerances select compensated dot accumulation and
recover the independently known exact small-integer answers.

All eleven original synthetic programs assembled for `sm_89` and passed their
independent references on the local GPU after explicit target adaptation.
All twelve additional semantic fixtures also assembled and executed correctly:
lane-indexed warp trees, explicit strided-loop bound bindings, uneven loop exits
before a shuffle, immutable snapshots around mutable state, fast exponential,
logical negation, erf, negative-base square/cube, half-precision selection, and
signed integer remainder. Together these provide 23 independent small programs.

The optional optimization tests add seven programs, checked in both widened
FP64 and compensated FP32 dot modes (14 numerical executions and 14 assembly
checks). They cover row and transposed layouts, adversarial cancellation,
carrier observations through `Select` and loop bounds, an ordinary update after
a dot loop, and a dot/shuffle/dot continuation. These tests found and now guard
against imprecise independence analysis and writes to temporary corrected
values instead of the underlying accumulator.

Seven more precision fixtures pass with both retained loops and unroll budget
128 (14 exact numerical comparisons and 14 assembly checks). They cover product
and sum residuals, `seed=False` across slabs, nested resets, snapshots across a
shuffle, half inputs with a float accumulator, standalone initial updates, and
an explicit base after a dot loop. These four fixture groups contain 37 distinct
small programs; selected modes add further executions of those cases.

The later `tests/test_layouts.py` suite adds eight distinct programs: both
orientations of grouped raster mapping with rectangular tails and batch axes,
an auxiliary producer warp across three CTAs, four lane-offset shared-copy
swizzles (`B32`, `B64`, `B128`, and `B128` with shift seven), and a warp-private
neighbor read after uneven loop work. All eight exact numerical comparisons
passed locally; all sixteen native/synchronous-copy variants assembled for
`sm_89`. The local swizzle checks use the explicit synchronous-copy alternative
on `sm_75`; they do not execute native `cp.async`. Evidence is in
`artifacts/validation/layouts/summary.json` and `layouts/unittest.log`, tied to
the historical backend hash above. Both layout methods also passed in the final
20-method run described above. The local totals exclude the separate
instrumented sanitizer executions.

```powershell
$env:EMMY_GPU_TESTS='1'
python -X utf8 -m unittest discover -s tests -p test_layouts.py -v
```

Compute Sanitizer `memcheck` completed with zero reported errors on four local
cases: the odd conditional/reordered-argument tile, segmented shared-memory
tree, masked-tail matmul, and fresh rectangular attention. All four also passed
their numerical references under instrumentation. The log and result report are
`artifacts/validation/memcheck.log` and `local-memcheck.json`.
These four runs used development revisions before the execution-gate source
freeze; they were not repeated for the final source hashes above.

A separate sanitizer suite ran five cases through each of `memcheck`,
`racecheck`, and `synccheck` (15 instrumented runs): accumulator snapshots,
pre-shuffle snapshots, uneven loop exits before a shuffle, lane-indexed warp
trees, and split-K matmul using the explicit synchronous-copy fallback. Every
run returned the correct result with zero reported memory/synchronization
errors and zero shared-memory hazards or warnings. Evidence is retained under
`artifacts/validation/sanitizer/`. These local checks do not certify native
`cp.async`, which requires newer hardware than the development GPU.

The new uneven-work warp-barrier fixture was additionally checked with
`memcheck`, `racecheck`, and `synccheck` on that backend revision. All
three instrumented executions returned exact numerical results with zero
errors, hazards, or warnings. Their commands, results, and logs are preserved
in `artifacts/validation/layouts/`. This is same-revision sanitizer evidence
for the corrected warp-barrier path; the earlier five-case sanitizer suite
remains development-revision evidence.

The ten fresh workload packages passed local execution and the final official
runner gate at seeds 0 and 17: pointwise length 257; nonlinear pointwise 17x65; sum length
1003; row sum 7x513; maximum reduction 9x73; softmax 7x93; RMSNorm 9x73; matmul
64x96 times 96x80; and attention with Q=16x16, pretransposed K=16x32, V=32x8.
The tenth, matmul 33x17 times 17x29, emits the `Select` statement that TASK.md
explicitly excludes from both public and hidden requirements. It initially
produced an unsupported-operation diagnostic; after adding ordered generic
`Select` lowering, this case also passed local and official execution at seeds
0 and 17.

An earlier strict-backend stress run used all-ones, alternating-sign, and normal
inputs scaled by eight at seed offset 101. **44 of 45 cases passed at that stage.**
The scaled-input 512x512
matmul had one mismatch out of 262,144 outputs: at `[278, 332]`, strict separate
FP32 multiply/add produced `0.3444877863` versus the FP64-derived reference
`0.3431081474`. Its absolute error `0.00137964` narrowly exceeds the specified
`0.00134311` tolerance at this cancellation-sensitive output. The ordinary
distribution and the all-ones/alternating inputs passed. This exploratory
failure remains visible in `artifacts/validation/remote-public-stress.json`.
The final compiler's compensated independent-dot policy resolved it: the same
wide-input case has zero mismatches out of 262,144 outputs and maximum absolute
error `7.62939453125e-06`. The checker uses the original tolerances throughout.
The ordinary-distribution baseline is retained in `remote-public.json`.

These results establish tested coverage, not hidden-set correctness or a claim
of winning the contest. Remote timing uses unlocked clocks and is indicative;
performance claims require repeated, interleaved comparisons on the same GPU.

## Historical clean installation and standalone CLI (23 October, phase 2)

The frozen backend
`0373bbc64340ae37274a67a92deb7320915ff4aa5cef97a231b3ac1ec7a2cc59`
passed a separate packaging check using a new CPython 3.12.7 virtual environment
with system site-packages disabled. Only `compile.py`, `ptx_backend.py`, and
`requirements.txt` were copied into the staged submission directory. Emmy was
imported from that environment's installed `emmy-ml==0.3.11`, and the backend
was imported from the staged copy. `pip check` reported no broken requirements.
The resolved NumPy and requests versions were 2.5.4 and 2.34.2; the complete
dependency inventory is retained in
[`installed-versions.txt`](../artifacts/validation/clean-install/installed-versions.txt).

Seven inputs covered the five public categories plus generated masked-tail
matmul and rectangular attention. Each IR was copied into an anonymized input
directory with only its launch/argument contract and numerical tolerances;
reference CUDA, expected outputs, input arrays, and expression generators were
absent. Each case passed two ordinary `python compile.py kernel.ir.json`
invocations and one guarded invocation of the same CLI: **21 successful
compilations**, with identical PTX bytes across all three runs per case and no
stderr output. Missing arguments, extra arguments, and malformed JSON also
passed three separate diagnostic checks: nonzero status, a useful stderr
message, and empty stdout.

The guarded executions rejected Python-observable network access, child-process
creation, CUDA library loading, calls to Emmy renderer functions, imports of
other compiler backends, and reads from the original source repository. All
seven completed without a prohibited event or renderer call. This supplements
source review; Python audit hooks are instrumentation, not an operating-system
sandbox against arbitrary native code. The installation step used the package
index; the subsequent compiler checks required no network or GPU access.

To reproduce from the repository, first generate the practice packages as
described above, then use a Python installation that supports virtual
environments:

```console
python tools/check_clean_install.py --install
python tools/check_clean_install.py
```

`--install` creates the environment if needed and installs the declared
requirements. Omit it to check an already installed environment without the
package-installation step. `--venv` selects another environment, and repeatable
`--case` arguments select a different set of input packages. This machine's
Anaconda installation lacked the standard `venv` module and produced a
virtualenv with an unavailable `_ctypes` DLL dependency, so the check used a
workspace-local standalone CPython and `.build/clean-env`. No global Python
installation was changed.

The final
[`summary.json`](../artifacts/validation/clean-install/summary.json) records the
source hashes, input and PTX hashes, module origins, observed file accesses,
interpreter configuration, timings, and diagnostic results. The staged source
snapshot and inputs remain under the workspace's `.build/clean-install-*`
directory. An earlier check passed its 21 compiler invocations but correctly
failed its final source-stability guard when the backend changed during the
run; its separate `prior-f7a7d5d-source-changed.json` is retained and excluded
from the final result. The final source hashes remained unchanged throughout.

This is same-host packaging and emission evidence, not an additional PTX
assembly or GPU execution result. Requirements pin Emmy but leave NumPy,
requests, and transitive dependencies open; the recorded inventory is an
environment snapshot, not a hash-locked dependency specification or proof of
compatibility with every future dependency release.

## Historical named-barrier instrumentation (23 October, phase 2)

The same frozen backend `0373bbc64340ae37274a67a92deb7320915ff4aa5cef97a231b3ac1ec7a2cc59`
passed **nine instrumented numerical executions**: three named-barrier fixtures
under each of Compute Sanitizer `memcheck`, `racecheck`, and `synccheck`.
Each execution returned the independently expected exact output. The memory and
synchronization checks reported zero errors; race checks reported zero hazards,
errors, and warnings. These are additional instrumented executions, not nine
additional distinct fixture programs.

The fixtures exercise two participating warps within a three-warp CTA, reuse of
a counted barrier inside nested loops with completion before shared-memory
overwrites, and a counted shared-memory reduction whose root is broadcast to
all participating lanes. The third warp follows a separate valid path. The
compiler preserves the explicit barrier identifier and participant count.

```console
python tools/check_sanitizer.py --suite contract-edges --adapt-local-target
python tools/check_sanitizer.py --suite contract-edges --list-cases
python tools/check_sanitizer.py --only named_barrier_nested_reuse --tools racecheck synccheck --adapt-local-target
```

The default selects the three named-barrier cases. `--only` accepts repeated or
comma-separated fixture names; `--sanitizer` selects an explicit executable.
Every execution requires a recognized clean sanitizer summary, a successful
worker exit, and a numerical comparison of every output. Missing summaries,
truncated outputs, infrastructure errors, timeouts, and source changes cannot
silently pass. The script serializes executions and uses a new run directory
so stale worker files cannot satisfy a later invocation.

The local GPU was a GTX 1650 Ti (`sm_75`); the explicit adaptation changed only
the emitted PTX target directive from `sm_89`. The
[`summary.json`](../artifacts/validation/contract-edge-sanitizer/summary.json)
records source and artifact hashes, original and loaded PTX hashes, complete
worker outputs, sanitizer summaries, commands, and target adaptation. The
referenced unique run directory retains inputs, expected arrays, emitted PTX,
requests, and complete logs. All tracked source hashes remained unchanged.
These fixtures contain no asynchronous copies. Their clean instrumentation
does not certify native `cp.async`, all future schedules, or kernel performance;
instrumented elapsed times are not benchmark measurements.
