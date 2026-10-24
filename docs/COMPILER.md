# Direct Emmy Kernel IR to PTX

This implementation walks typed Emmy 0.3.11 statements and emits PTX 8.0 for
`sm_89`, selecting `sm_90` when bulk-copy/barrier nodes require it. It does not inspect problem filenames, expression descriptions, CUDA
references or expected outputs. The submitted path uses no CUDA compiler,
Triton, LLVM, NVVM or Emmy renderer. `ptxas` remains the downstream assembler.

```console
python -m pip install -r requirements.txt
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
python submit.py p00_scalar_add --compiler compile.py
```

Python 3.12+ is required by Emmy. The input's sibling `meta.json` supplies the
parameter order and dynamic shared-memory contract. Standard output contains
only ASCII PTX; diagnostics go to standard error with a nonzero exit status.
The compiler itself needs neither a GPU nor a network connection.

## Architecture

`compile.py` is the submission interface. `ptx_backend.py` contains the core
backend; `ptx_tensor.py` and `ptx_bulk.py` isolate optional hardware instruction
families. Its typed `Value` records track PTX register types and proven integer
divisibility; `Buffer` records track shape, element type, address space and
alignment. There is no physical register allocator: fresh virtual registers
keep instruction generation straightforward and `ptxas` allocates hardware
registers.

The lowering proceeds through these stages:

1. Deserialize with `Graph.from_dict`, identify one `KernelOp`, and validate the
   exact entry name and metadata argument set.
2. Collect buffer layouts and aligned shared-memory declarations. Kernels above
   the static 48 KiB allocation limit use one dynamic shared pool; the compiler
   checks that the prescribed launch supplies enough bytes.
3. Load pointer parameters, establish thread/lane/warp bindings, and recursively
   translate expressions and statements. Row-major offsets use buffer shapes;
   large layouts widen before multiplying index terms.
4. Emit the entry signature, register declarations, shared storage and code.

The compiler supports every required statement kind: `Tile`, `Load`, `Write`,
`Assign`, `Let`, `Init`, `Accum`, `Loop`, `StridedLoop`, `Cond`, `Smem`, `Sync`,
`WarpShuffle`, `TreeHalve`, and the three `CpAsync` operations. It also supports
ordered `Select` branches, which help with freshly generated masked-tail
matmuls although the competition excludes them from required coverage.

Optional explicit tensor IR lowers to m16n8k16 FP16/BF16-to-FP32 MMA, shared
`ldmatrix` or guarded global gathers, fragment families and scalar epilogues.
Bulk IR lowers in-bounds unswizzled boxes to aligned `cp.async.bulk` rows and
transaction barriers without adding host tensor-map arguments. The
[feature index](FEATURES.md) specifies the supported forms, architecture checks,
examples, negative tests and remaining IR preconditions. These extensions are
selected by node type and leave ordinary scalar scheduling unchanged.

## State and synchronization

Each loop seeds only its immediate accumulators. `seed=False` updates an
existing carrier across slab iterations. `Accum.base` supplies the actual left
operand of an update; it is essential to the rescaling in online softmax.
Register snapshots preserve an SSA value copied before a mutable carrier
changes. Nested scopes restore value bindings without losing the enclosing
loop's running state.

Warp butterflies shuffle every component before running the ordered merge
program. Participation masks come from complete aligned subgroups described by
the IR, so uneven loop trip counts cannot turn a transient active mask into a
partial reduction. A shared tree runs the same merge program, retains each
barrier, and broadcasts every state component to the consumers. Lane-indexed
trees preserve all inner columns. The scalar per-warp tree uses one writer per
slot to avoid redundant same-address shared writes.

Global-to-shared copies retain their commit groups and waits, including empty
groups. Copy completion and consumer barriers remain distinct. Split-K outputs
use atomic addition; the launcher performs the required output reset before
every invocation. Kernels never attempt an unsafe inter-block zeroing phase.

## Numeric and optimization policies

Ordinary floating-point add, subtract and multiply use explicit round-to-nearest
instructions. Vector operations are emitted only when their address alignment
is proven; otherwise the same consecutive elements use scalar memory operations.
Constants passed as buffer parameters remain loads, rather than being replaced
by presumed input data.

The file-level default uses compensated FP32 for independent additive dots when
metadata specifies nonzero `rtol` or `atol`. Exact or missing tolerances retain
ordinary arithmetic. Explicit API flags select strict arithmetic, adjacent FMA
contraction, FP64 accumulation, or compensated FP32 for development comparisons.
The lower-level `Compiler` constructor defaults to strict arithmetic.
Eligibility comes from the IR dataflow: an adjacent product feeds
an additive accumulator without an explicit rescaled base, and the carrier has
no in-loop consumer. Coupled normalization state and observed recurrences are
excluded. Tests exercise both positive and negative recognition cases.

Compensated accumulation uses a product residual from FMA and the full TwoSum
transformation, following Dot2 in Ogita, Rump and Oishi's
[Accurate Sum and Dot Product](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).
Both the main sum and correction reset with the original carrier and survive
`seed=False` loops. Consumers receive their rounded sum. At each consumer, a
nonfinite main sum selects a zero correction and still executes `high + 0`.
This preserves the existing infinity/NaN propagation while moving its guard
out of the per-product update. Between resets, the main sum changes only by
addition, so it cannot become finite again after becoming nonfinite; every
replacement clears or recreates the correction. The finite arithmetic order
and numerical policy are unchanged by this guard movement. Compensation improves
accuracy against the mathematical reference; it is not a claim of bitwise
equivalence to separately rounded FP32 accumulation or exact arithmetic for
arbitrarily ill-conditioned inputs.

Scheduled unrolling is bounded by a structural code-size estimate: 1,024 per
loop and 4,096 across the kernel. Large outer attention loops remain loops.
There is no online training, filename lookup,
output cache, network evaluation, or autotuning in the submitted compiler.
`tools/tune_lowering.py` performs reproducible development comparisons with
independent correctness checks, rejected candidates and interleaved timing.
The [performance study](PERFORMANCE.md) records the numerical improvement and
the attention runtime cost of the selected default.

## Evidence and practical limits

See [VALIDATION.md](VALIDATION.md) for recorded results and reproduction commands,
[SEMANTICS.md](SEMANTICS.md) for the node-by-node interpretation, and
[VIDEO_GUIDE.md](VIDEO_GUIDE.md) for a walkthrough to prepare in your own words.

The demonstrated workload families are FP32 pointwise, reductions, matmul,
normalization, softmax and attention. Smaller tests additionally exercise
integer operations and narrow floating-point arithmetic. This is not a complete PTX
implementation of every NumPy operation or a correctly rounded math library.
Unknown operation names, unsupported type pairs and incorrect arity fail
explicitly before stdout is written. Direct BF16 `BinaryExpr` arithmetic and
FP16 expression division now use the documented FP32 computation and explicit
narrow-rounding path. Each narrow binary node retains its rounding boundary.
The [type capability map](TYPE_SUPPORT.md) details supported node forms,
target-dependent conversion limits, raw-bit handling and signed population-count
semantics. Nineteen numerical packages exercise these paths; this remains a
bounded scalar implementation, not exhaustive datatype/domain coverage.
Transcendentals use documented
PTX approximations or an explicitly documented approximation; their full-range
error is not certified. Optional tensor cores and restricted bulk transfers are
now implemented under the [feature contracts](FEATURES.md). General host-encoded
TMA, dynamic dimensions and serial runtime axes remain unsupported.

The [transformation contracts](TRANSFORMATION_CONTRACTS.md) state scope,
guard, rounding and effect preconditions. `tools/analyze_ir.py` exports structural
features and optional emission/assembly statistics without launching a GPU or
altering the production compiler. Access-site bytes are not measured memory
traffic, and virtual registers are not physical allocation.

The public tests and generated counterexamples provide evidence of generality,
not access to the hidden tests or a promise of winning. The existing user's
README edit is preserved; no repository has been published or entry submitted.
