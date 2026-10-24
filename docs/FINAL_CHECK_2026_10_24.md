# Public-client rerun and fresh typed-storage checks

This is additional development evidence for the existing compiler, dated
24 October 2026. Passing the public set is the baseline. The purpose of the
new cases is to exercise different shapes, operations and actual buffer types;
they are not the organizer's unavailable hidden set.

## Unmodified public client

All **15/15 public problems pass** through the repository's unmodified client:

```console
python submit.py <problem> --compiler compile.py
```

The invocation was executed once for every public problem, serially. The
[closed report](../artifacts/validation/2026-10-24-public-submit-runner/summary.json)
retains all commands, exit codes and stdout/stderr hashes. Every expected output
has a corresponding PASS line. Compiler, client, metadata, IR and reference-file
hashes are unchanged before and after execution. The report SHA-256 is
`6d521b3f1bb45bd238a4a4ee6fa940fa40f374b9d8bea0be4f443b5559c872b2`.

To repeat this exact public-client workflow and retain a fresh report:

```console
python tools/check_public_submit.py --report-dir artifacts/validation/my-public-rerun
```

The report directory must be new. The initial sandbox-restricted attempt reached
no GPU executions; all 15 commands ended with ConnectionError. Its logs and
[classification](../artifacts/validation/2026-10-24-public-submit/classification.json)
remain separate. The authorized retry outside that network restriction is the
successful run above. No compiler failure is inferred from the connection errors.

## New corpus and validation method

[expressions_oct24.py](../tests/expressions_oct24.py) defines 18 planned workloads
in all five categories. The expressions and tolerances were fixed before
compiler feedback. All **54 CPU PyTorch versus independent NumPy/Python reference
checks pass**: normal inputs with seed offsets 0 and 17, and wider inputs with
offset 101. The [protocol](../artifacts/validation/2026-10-24-extra/protocol.json)
and [reference check](../artifacts/validation/2026-10-24-extra/reference-sanity.json)
retain this preparation evidence.

The repository's `tools/generate.py` performs tracing and packaging. On Windows,
`tests/generate_stress.py` supplies its existing import-only `fcntl` shim and
isolated tuning-file paths; it calls the same generator functions. This is test
preparation, separate from the submitted IR-to-PTX translation. Torch and Emmy's
CUDA renderer are not runtime dependencies of the submitted compiler.

The planned cases include real FP16, FP64 and I32 buffers, odd row strides and
tail dimensions, FP64 reductions and contractions, signed integer selection and
bit operations, trigonometric/error functions, normalization and attention.
Preparation checks include reference/input ordering and generated output dtype.
Frontend failures are reported separately from emitter or execution failures.

The stock `submit.py` reconstructs returned values as FP32. It is appropriate
for the public FP32 set, but cannot establish FP64 precision or large-I32
correctness. Additional cases therefore use the dtype-aware
`tools/check_compiler.py` comparison path and retain request/response evidence.
No local GPU execution is used.

The original generator produced 16 packages and rejected two expressions in its
CUDA-reference renderer (`ceil` and `remainder`). A separate input audit rejected
the original normalization package because it captured intermediate `z` instead
of source input `x`. An IR-vocabulary audit rejected the original FP16 matmul's
excluded `MmaSyncPtx` schedule. The original files and failed audits remain
archived. Separately named inline-normalization and scalar-matmul variants retain
the mathematics, shapes, references and tolerances. Their six additional CPU
checks bring the generated-corpus CPU reference ledger to **60 checks**, not 60
different programs. The [final preparation roster](../artifacts/validation/2026-10-24-extra/final-admission.json)
records all four rejected originals and both repairs.

The initially admitted 16 graphs have five FP16, four FP64, three I32 and four
FP32 input-buffer cases. The
[ABI/vocabulary audit](../artifacts/validation/2026-10-24-extra/input-audit-final.json)
passed, but did not establish the precision of internal accumulation. That
distinction matters in the observed failures below.

| Executed set | Mathematical-reference result | Evidence |
|---|---:|---|
| Public client rerun | 15/15 passed | [Unmodified-client report](../artifacts/validation/2026-10-24-public-submit-runner/summary.json) |
| First four generated packages, three inputs each | 12/12 passed | [Early report and raw evidence](../artifacts/validation/2026-10-24-root-audit/early-remote/summary.json) |
| Remaining twelve generated packages plus one explicit large-I32 input | 28/37 passed; nine failures retained | [Remaining report](../artifacts/validation/2026-10-24-extra/remote/summary.json) |
| Two hand-built typed-IR operation fixtures | 6/6 passed | [Operation fixture report](../artifacts/validation/2026-10-24-root-audit/typed-extra/remote/summary.json) |
| Three explicit FP64-state test variants | 9/9 passed | [FP64-state report](../artifacts/validation/2026-10-24-root-audit/fp64-state-variants/remote/summary.json) |

This turn executed **79 workload/input checks: 70 matched their mathematical
references and nine did not**. Outside the public set, the accounting is **64
checks: 55 mathematical-reference passes and nine retained failures**. The
previous 332-check release history is not added to these counts. Repeated
launches, CPU attribution and preparation attempts are not new GPU checks.

### Why nine generated FP64 comparisons failed

The generated column sum, matmul and inline normalization have FP64 buffers, but
their Kernel IR explicitly declares FP32 accumulators and cooperative state.
Normalization also declares FP32 reciprocal/final multiplication. All three
ultimately store FP32-valued results widened to FP64. Their tight FP64
mathematical references therefore do not match the scheduled arithmetic.

All nine raw GPU outputs match independent CPU implementations of that exact
declared schedule **bit-for-bit**, across 3,057 output elements. For **3,052 of
those elements**, even the nearest FP32 value widened to FP64 lies outside the
original FP64 tolerance. Every failed case contains such elements. This provides
an independent precision-incompatibility check, not merely an explanation based
on the size of the observed error. The
[attribution report](../artifacts/validation/2026-10-24-extra/precision-attribution/summary.json)
retains inputs, simulated results, raw-response identities and original failure
flags. It performed no new GPU executions. The
[raw-integrity audit](../artifacts/validation/2026-10-24-extra/remote/integrity-audit.json)
separately verifies the 37-case roster and all 62 request/response events; its
numerical gate remains explicitly false at 28/37.

The compiler was **not** changed to infer arithmetic precision from a buffer's
dtype. Such a rule would also erase intentional mixed-precision calculations.
The original IR, expected outputs and tolerances were not edited, and its nine
failed comparisons have not been relabeled as passes.

### Additional valid typed-state coverage

The two [operation fixtures](../tests/test_oct24_elementwise.py) independently
construct FP64 floor/ceil/trunc and FP32 remainder Kernel IR, because the
generator's CUDA-reference renderer rejected those operations. Their six GPU
checks, six CPU-reference checks and two assemblies passed. They are explicitly
labeled **hand-built IR**, not generator outputs.

The three [FP64-state variants](../tests/test_oct24_fp64_states.py) parse the
original typed graphs and explicitly choose FP64 local/cooperative state and
shared storage. The matmul's optional FP32 `Select` becomes a typed FP64 ternary
`Let`. Global buffer indexing, logical shared indices, launch geometry,
predicates and synchronization are preserved; scratch storage is widened and
aligned for double precision. Expected NPZ files are copied byte-for-byte, and
the original tight tolerances remain unchanged. These are **test-only IR
variants**, not a new production compiler policy or unchanged generator output.
All nine CPU reference checks, three assemblies and nine GPU checks pass. The
[variant manifest](../artifacts/validation/2026-10-24-root-audit/fp64-state-variants/manifest.json)
and [raw audit](../artifacts/validation/2026-10-24-root-audit/fp64-state-variants/remote/integrity-audit.json)
verify that their PTX contains no FP32 instructions and retain the transformation
and original-package identities.

The [large-I32 case](../artifacts/validation/2026-10-24-extra/explicit-large-i32/manifest.json)
is one additional input for an existing generated selection graph. Its kernel
IR/PTX remain unchanged. It passes exact comparison for 1,043 outputs, of which
625 would change under an FP32 round trip. It is counted once, not as a new
generated shape or three different seeded cases.

The [final evidence index](../artifacts/validation/2026-10-24-final/summary.json)
checks source identities, raw-artifact hashes and exact disjoint case rosters.
Its release-review status means the compatible-reference gates passed and the
nine incompatible-reference failures were fully attributed; it does **not**
mean that all 79 original mathematical comparisons passed.

FP16 matrix multiplication uses the scalar-schedule variant and is checked
against its declared `rtol=0.01, atol=0.01`. This is tolerance-level evidence,
not a claim of correctly rounded native FP16 GEMM. Tight FP64 cases check a
different numerical requirement. Timings from correctness launches are not
paired performance benchmarks.

## Review and presentation changes

The [README architecture map](../README.md#compiler-architecture) points to the
actual ABI, statement, expression, memory and cooperative-lowering functions.
The [video script](VIDEO_SCRIPT.md) contains 617 spoken words, approximately
4-5 minutes, with separate screen cues. It explains Tile and StridedLoop,
shuffle/tree state combination and broadcast, asynchronous-copy grouping,
tolerance-gated independent-dot compensation, and architecture-dependent BF16
conversion. The source review found no demonstrated defect in those mechanisms
that justified changing the tested implementation merely to shorten it.

[SEMANTICS.md](SEMANTICS.md) also corrects an outdated narrow-type limitation:
the phase-4 implementation already handles FP16 division and BF16 binary
arithmetic with explicit per-node narrowing. The detailed boundaries remain in
[TYPE_SUPPORT.md](TYPE_SUPPORT.md).

## Submission identity

The production files remain the tested phase-4 files:

| File | SHA-256 |
|---|---|
| `compile.py` | `4ed3fb67833b7de20693c7cb3f163a4646cac8d3a71e63649c1af1bccd37ce56` |
| `ptx_backend.py` | `5cb3a351840ca03055b9725471fd3b44afee7a1d8e2832c49d439018da5ba9fc` |
| `requirements.txt` | `5d054f8757b1212865653105a4e7c634e2952007da020d2f67ad456eb6bccc84` |

The [submission ZIP](../artifacts/release/phase4/submission-bcbc4a8381497f1d34be.zip)
contains exactly these files and was independently checked byte-for-byte against
the current source. Its SHA-256 remains
`b9af6a761233e39b8faa330e1fd01c49e8a457498cb5d3b7e2465ef5403251e8`.
Documentation and additional development tests do not change this minimal
compiler archive. Historical reports and the dated pipeline PDF remain intact.

The refreshed [release manifest](../artifacts/release/phase4/submission-bcbc4a8381497f1d34be-3908fd80497591da4d12.manifest.json)
binds the original five passing release reports and today's complete review
index, and records another successful extracted-CLI check. Its SHA-256 is
`3908fd80497591da4d12af75b4203bbffeceb5c3372d1a046e02450a217a95ec`.
The ZIP bytes are unchanged; the new manifest adds evidence without silently
replacing the old reports or claiming a compiler improvement.

The package is prepared locally. Recording the user's own video, publishing
the repository/video and submitting the formal entry are separate delivery
steps; this validation work does not perform those actions.
