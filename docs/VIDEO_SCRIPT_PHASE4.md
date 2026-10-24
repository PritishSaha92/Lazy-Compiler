# Read-aloud compiler walkthrough

Read only the text between the two markers. The spoken text contains
617 words, approximately 4-5 minutes at an unhurried technical speaking pace. The screen plan and code anchors below
are production notes, not spoken text. The script describes the implemented
backend without claiming a hidden-set result or a speed ranking.

<!-- SPOKEN SCRIPT START -->

I'm presenting a Python compiler that translates Emmy's scheduled Kernel IR
directly into PTX. I used AI assistance during development. I'll explain the
lowering choices and the evidence behind them.

The command takes an existing IR JSON file. Emmy deserializes it into typed
objects, and the neighboring metadata supplies the argument order, launch
contract and tolerances. The backend walks those objects and emits PTX itself.
It does not compile CUDA, read expected outputs, or recognize public problem
names. NVIDIA's assembler handles physical register allocation afterward.

First, Tile describes work ownership. I derive a linear work-cell index from
the block and thread IDs, account for auxiliary threads, reject out-of-range
cells, then decode the scheduled axes and raster order. This preserves the
schedule rather than inventing a new grid.

StridedLoop becomes an induction register, a bound check, a body and a back
edge. Its start and explicit end are evaluated on entry; its step advances the
index. Accumulators declared directly in that loop are initialized only when
seed is enabled. A seed-false loop continues existing state, which matters
when a matrix product accumulates across several slabs. Nested reductions reset
at their own nesting level.

Next, reductions have two communication paths. WarpShuffle uses XOR exchanges
inside the IR's aligned power-of-two subgroup. The participation mask comes
from that subgroup, not a momentary active-lane snapshot. At each stage, all
components are exchanged before the ordered state-combination program runs.
This matters for softmax, where the maximum and normalizer are coupled.

TreeHalve combines partial states through shared memory. Eligible workers merge
pairs, store the result, and participating threads reach the specified barrier
before the next stage. After the tree, each consumer loads the final reduced
slot. That broadcast is essential when normalization needs the result beyond
thread zero.

For asynchronous copies, the backend emits global-to-shared cp.async
instructions with the IR's addresses and byte counts. Commit closes a copy
group. Wait-group preserves the requested number of recent groups that may
remain pending; it does not always drain everything. Copy completion and
consumer synchronization are separate, so the following IR barrier is
preserved. This keeps the software pipeline's buffer reuse correct.

Numerics are an explicit policy. A positive relative or absolute tolerance
enables compensation only for structurally eligible independent FP32 dots.
The detector requires adjacent multiply-and-add updates and rejects a carrier
read during accumulation, including reads in conditions or loop bounds. Exact
or missing tolerances keep ordinary FP32 accumulation. Coupled online
normalization also keeps its ordinary lowering.

For an eligible dot, TwoProductFMA keeps the rounded product and recovers its
residual with a fused multiply-add. Full TwoSum recovers the error from adding
that product to the high accumulator. Both errors accumulate in a low
component, which is included when the result is consumed. Both components
follow the same seed and reset rules. This improves the tested cancellation
cases, but changes sequential rounding; a nonzero tolerance is not a general
license to rewrite arithmetic.

BF16 conversions are architecture-aware. Below SM80 they are rejected. On
SM80 and newer, BF16 expands exactly to FP32. FP16 and small integer sources
also expand exactly before rounding to BF16. FP64 and 32- or 64-bit integer
sources require SM90's direct conversion; earlier targets reject them rather
than silently double-rounding through FP32. BF16 arithmetic and comparisons
use FP32 expansion with explicit result rounding where required.

Finally, all fifteen public problems pass the unchanged runner client. The
extra tests revealed an important distinction: FP64 buffers can still have
FP32 accumulators in the generated IR. Nine mathematical-reference checks fail
for that reason, while their outputs match the IR bit-for-bit. Separately
constructed FP64-state fixtures pass the same original tolerances. Both results
are retained. The README maps these mechanisms to code; tests provide evidence,
not a guaranteed competition result.

<!-- SPOKEN SCRIPT END -->

## Screen plan (do not read aloud)

| Script segment | Show |
|---|---|
| Contract | `compile.py`; `compile_file`; one real public IR path and its neighboring `meta.json`. |
| Work ownership and loops | `Compiler.tile`, `Compiler.loop`; point to the guard and `seed` check. |
| Cooperative reductions | `Compiler.shuffle`, `Compiler.merge`, `Compiler.tree`; show exchange-before-merge and the final shared load. |
| Async pipeline | `Compiler.statement` branches for `CpAsyncCopy`, `CpAsyncCommit`, `CpAsyncWait`, plus the IR's separate `Sync`. |
| Numerical policy | `compile_file`, `Compiler.independent_dots`, `Compiler.compensated_dot`, `seed`, `lookup`, `write_state`. |
| BF16 | `Compiler.convert`; show the SM80/SM90 checks and exact-intermediate whitelist. |
| Evidence | [24 October results and case accounting](FINAL_CHECK_2026_10_24.md), [public-client report](../artifacts/validation/2026-10-24-public-submit-runner/summary.json), and [full review index](../artifacts/validation/2026-10-24-final/summary.json). Keep original failures and separately constructed FP64-state fixtures visible. |

A concrete live command is:

```bash
python submit.py p00_scalar_add --compiler compile.py
```

For a PTX emission demo in Windows PowerShell 5.1:

```powershell
cmd /c "python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx"
```

## Source accuracy notes (do not read aloud)

- `Tile` emits a cell guard; the input schedule must still satisfy its original
  barrier-participation contract. This is not a claim that arbitrary divergent
  barriers become valid.
- `TreeHalve` honors named barrier IDs/counts and interleaved inner slots. A
  warp-indexed tree without an inner slot uses one representative lane to avoid
  redundant same-address stores; all required participants still synchronize.
- Compensation is not global FMA contraction or a different launch schedule.
  The high component's nonfinite guard is applied when the correction is
  materialized. The implementation does not claim universal correct rounding.
- Architecture here means the **emitted PTX target**, not the newer physical GPU
  executing it. A module targeting `sm_89` retains its SM89 conversion rules on
  the SM120 runner. BF16-to-other-format conversions use an exact FP32 expansion
  even when targeting SM90.
  Small integer sources mean signed/unsigned 8- and 16-bit values; predicates
  use exact zero/one. Unsupported type/operator pairs fail explicitly.
- Compilation uses no test input data. The development runner necessarily reads
  inputs and expected outputs to validate the generated kernel.
- Do not claim to have memorized or personally derived a mechanism merely by
  reading this script. The code anchors make each explanation inspectable.
