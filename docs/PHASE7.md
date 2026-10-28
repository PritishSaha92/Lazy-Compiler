# Integer semantics and logical GPU builtins

**Validated release: 28 October 2026.** Three correctness fixes passed fresh
organizer-GPU checks: all 15 public problems and 41 additional packages. The
additional checks comprise 29 existing feature examples, 11 new counterexamples
and one retained scalar case whose PTX changed. The [source-bound release
summary](../artifacts/validation/phase7/release-summary.json) connects the gates
below to the five runtime files. The [phase-6 release](PHASE6.md) and its evidence
remain historical and unchanged.

## Three correctness fixes

1. **GPU builtin names and types.** Emmy 0.3.11 uses logical names such as
   `thread_idx.x`, `block_idx.x`, `block_dim.x`, `grid_dim.x` and `warp_size`.
   The backend now resolves every x/y/z component and retains the corresponding
   CUDA-spelling aliases. Geometry components carry unsigned 32-bit types;
   `warp_size` is a signed 32-bit value of 32. This matters for comparisons
   against negative integers, arithmetic crossing the high bit, and shifts.
   Bulk-copy address proofs use the same name resolver and metadata bounds.
   Unknown names and unproved unsigned wraparound in bulk indices are rejected.
2. **Unstamped mixed-integer `Assign`.** For the supported integer arithmetic
   and bitwise operations, an omitted result dtype now follows Emmy's integer
   rank/sign rule instead of falling back to FP32. For example,
   `i16(1) + i32(16777217)` remains integer arithmetic and produces `16777218`.
   A floating intermediate loses information at this magnitude. Explicit
   result dtypes, floating defaults and the existing logical-mask policy retain
   their previous behavior.
3. **Expression shift promotion.** `BinaryExpr` shifts promote the operands
   independently and take the result type from the promoted left operand.
   A wide shift count no longer widens a 32-bit left operand, and an unsigned
   count no longer turns a signed arithmetic right shift into a logical shift.
   The PTX count is converted separately to `u32`. Constant folding and emitted
   instructions are checked against the same independent expected values, with
   counts inside the promoted left operand's width. Named operations with an
   explicit result dtype retain that contract.

The changes are in [ptx_backend.py](../ptx_backend.py) and
[ptx_bulk.py](../ptx_bulk.py). They preserve the input ABI, scheduled work,
launch geometry, floating-point policy and synchronization. They introduce no
new scheduling policy, trained selector or performance claim.

## Auxiliary threads and launch limits

`Tile.aux_threads` already had coordinate decoding in the backend. The new
builtin support makes explicit raw-thread role guards usable with Emmy's
logical names. The auxiliary fixture launches 64 compute threads and 32 helper
threads per CTA. Helpers populate shared memory, a CTA barrier publishes it,
and the compute threads use a counted 64-thread barrier while rearranging the
data. Three CTAs have independent inputs and outputs.

This is support for the explicit tested IR protocol. It does not infer thread
roles or missing barriers, and it does not add `SetMaxNReg`, WGMMA, or general
host-encoded tensor-map support. The existing tensor and bulk limits remain in
[FEATURES.md](FEATURES.md).

The direct builtin-value fixture deliberately uses a three-dimensional launch
**without `Tile`**. It verifies every component against the physical launch.
`Tile` continues to use the existing one-dimensional thread/block mapping;
general three-dimensional Tile schedules are outside this implementation's
contract. Tensor and bulk paths retain their one-dimensional launch checks.

## New independent examples

The [integer tests](../tests/test_unstamped_integer.py) define seven packages:
three mixed-integer arithmetic/bitwise cases, three shift cases with matching
constant/runtime outputs, and one explicit-dtype control case. Values above
the FP32 exact-integer range expose accidental floating conversions. Signed
runner buffers carry the inputs; unsigned and narrow operand types are explicit
IR conversions. Expected results are fixed integer examples.

The [builtin and auxiliary tests](../tests/test_builtin_aux.py) define four
packages: all builtin values in a direct 3-D launch, the auxiliary-thread
protocol, multi-CTA bulk coordinates, and unsigned builtin arithmetic.
The last fixture returns zero/one observations so FP32 output transport cannot
hide high-bit integer errors. Canonical and legacy names must emit identical
PTX. Negative tests retain unknown-name, out-of-bounds grid and unsigned-index
proof diagnostics.

These **eleven new packages** are development counterexamples, not organizer
hidden tests. Their inputs and expected arrays are included under
[examples/phase7](../examples/phase7). A twelfth package retains the existing
`comparison_integer_promotion` example unchanged: its shift-count instruction
type changed, so it received a fresh GPU check too. CPU/assembly checks and
organizer GPU execution are reported separately below.

## Preserved pre-fix evidence

The [six-package phase-6 baseline run](../artifacts/validation/phase7/pre-fix-integer-runner/summary.json)
is deliberately failing. It records three numerical mismatches, one compilation
failure for the mixed-integer XOR, and two output-decoding overflow errors after
the runner returned very large values for the signed-shift cases. The retained
wire exchanges distinguish those outcomes; the decoding errors are not
relabelled as completed numerical comparisons.

The additional builtin defects were found through rejected logical names and
incorrect signed expression types. The new fixtures and unsigned arithmetic
controls document the expected behavior. The [baseline name
reproduction](../artifacts/validation/phase7/before-fix/builtin-name-reproduction.json)
and [baseline runtime hashes](../artifacts/validation/phase7/before-fix/runtime-manifest.json)
retain the earlier diagnostics and source identity.

The nine older FP64-buffer/FP32-state mathematical-reference failures remain
failures under their original contracts. These integer/builtin changes do not
claim to fix them; see [the earlier review](FINAL_CHECK_2026_10_24.md).

## Release validation

| Gate | Status |
|---|---|
| [Unmodified public submission client](../artifacts/validation/phase7/public-submit/summary.json) | 15/15 passed |
| [Previous optional-feature corpus](../artifacts/validation/phase7/feature-runner/summary.json) | 29/29 passed |
| [New exact-reference packages](../artifacts/validation/phase7/new-feature-runner/summary.json) | 11/11 passed |
| [Retained scalar case with changed PTX](../artifacts/validation/phase7/changed-scalar-runner/summary.json) | 1/1 passed |
| [Complete CPU/assembler discovery](../artifacts/validation/phase7/cpu-regressions-final.json) | 129 methods: 110 passed, 19 explicitly skipped local-GPU methods |
| [Retained PTX compatibility](../artifacts/validation/phase7/compatibility.json) | 119/120 scalar modules and 29/29 previous feature modules byte-identical; the one changed scalar module passed the fresh GPU check |
| [Isolated installed-dependency check](../artifacts/validation/phase7/clean-install/summary.json) | 25 packages, 75 successful compilations and three expected diagnostics |

The 56 GPU checks use the organizer hardware; no local GPU was used. Independent
wire audits replayed all 41 additional outputs and checked their recorded
requests, responses and source hashes. PTX identity is a compatibility check,
not a substitute claim that all 120 retained scalar inputs were executed again.
The changed scalar output differs only in a shift-count `selp` changing from
`s32` to `u32`; its selected counts are zero and one.

The [first full CPU run](../artifacts/validation/phase7/cpu-regressions.json)
retains two failed diagnostic-message assertions. Invalid floating-point shifts
now report that shifts require integer operands; the old assertions expected a
generic binary-operation message. Only those expectations were updated before
the passing rerun. Runtime hashes remained unchanged throughout these gates.

| Runtime file | SHA-256 |
|---|---|
| `compile.py` | `4ed3fb67833b7de20693c7cb3f163a4646cac8d3a71e63649c1af1bccd37ce56` |
| `ptx_backend.py` | `3344668fe0b3f94a7198478fcccc50a51e80d70a33c67a868942353918dc9d4f` |
| `ptx_tensor.py` | `5d97c908dc0f397401cbe7894343dd1bbc3a2d8d3b0827fe650ccb3a3015e935` |
| `ptx_bulk.py` | `a906271f8c04a21b75c8796a65225a832b601807156d5406320b9c1b629de588` |
| `requirements.txt` | `5d054f8757b1212865653105a4e7c634e2952007da020d2f67ad456eb6bccc84` |

These results establish the tested behavior of this release. They do not claim
hidden-set results, a measured speed improvement, or a competition ranking.

## Recorded video and submission

The recorded two-screen video explains the architecture and the phase-6
validation snapshot: 15 public and 29 additional feature packages. Those are
historical recorded results, not a count of the new phase-7 run. The core
architecture, floating numerical policy, tensor/bulk extensions and evidence
boundaries described in the video remain applicable. This document supplies
the subsequent correctness changes and their separate validation evidence.

The [runtime archive pointer](../artifacts/release/phase7/latest.json) identifies
the five-file runtime package. The [review archive
pointer](../artifacts/release/phase7/review/latest.json) identifies the source,
examples, tests, documentation and selected evidence bundle. Use the review
bundle when updating a repository for manual inspection. Distribution pointers
are external release files; the review ZIP does not embed itself.

This revision is prepared locally. Packaging does not update the public GitHub
repository or resubmit the competition entry. The existing recorded video
remains a phase-6 explanation, with this page as the subsequent release addendum.
