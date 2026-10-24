# Kernel IR lowering notes

These notes were checked against every open problem's readable IR, typed
`Graph.from_dict` objects, CUDA reference, and metadata, using **emmy-ml 0.3.11**.
The installed package is authoritative for deserialization; the sibling Emmy
checkout can be newer. The submitted lowerer must emit PTX itself and must not
call Emmy's renderer, NVCC, or another compiler backend.

The optional tensor-core and bulk-copy node families now have separate
implementations in `ptx_tensor.py` and `ptx_bulk.py`. Their exact semantic
boundaries, lane layouts, bounds proofs and synchronization preconditions are
recorded in [FEATURES.md](FEATURES.md). The original required vocabulary below
remains unchanged; "not required" in the starter specification does not mean
that these optional families cannot be implemented.

## Contract and indexing

- Find `KernelOp` by type. Preserve `KernelOp.name` as the PTX entry name.
- Find the matching entry in adjacent `meta.json`; use its `arg_order`, including
  scalar constants represented by one-element pointer arguments. Graph input
  order is not the ABI. All open arguments are FP32 buffers passed as `.u64`.
- The runner owns the launch and zeroes the buffers listed in `zero_outputs`.
  An atomic kernel must not race to zero its own output.
- A Tile's cell count is the product of its axis extents. Normally
  `gid = blockIdx.x * blockDim.x + threadIdx.x`. Guard `gid < cells`, then decode
  each axis in row-major order: `(gid / product(inner_extents)) % extent`.
  The outermost modulo is redundant. Do not infer axis coordinates from names.
- With auxiliary producer threads, `gid = blockIdx.x * block_threads +
  threadIdx.x % block_threads`. The raw thread ID still defines `lane = tid & 31`
  and `warp = tid >> 5`.
- Flatten an index against its buffer shape. An empty index is zero; a single
  index is already flat. A matching multidimensional index uses row-major
  strides. Emmy's legacy rank-mismatch behavior sums the supplied indices.
  `test_index_layout.py` covers rank-two coordinates on rank-one and rank-three
  inputs, rank-mismatched output coordinates, ordinary row-major coordinates,
  and empty/single indices. Compile-time checks retain 64-bit offsets without
  allocating oversized buffers on the test GPU.
- Widen address products before multiplication when sizes can exceed 32-bit
  range. Widening only the final sum preserves an already-overflowed result.
- Raster mapping is a permutation of the specified pair of block axes. For
  grouped-axis extent `A`, other extent `B`, group `G`, and flat pair subindex
  `p`, define `group = p // (G*B)`, `size = min(G, A-group*G)`,
  `a = group*G + (p % (G*B)) % size`, `b = (p % (G*B)) // size`.
  `raster_orient` determines which of the pair is `a`; preserve outer axes and
  inner thread axes. `a02` exercises raster metadata even though its second
  block axis has extent one.

## Integer expression policy

Integer `BinaryExpr` arithmetic follows the installed CUDA rendering convention:
`/` and `//` become C integer division, truncating toward zero; `%` retains the
dividend's sign. Constant folding and register arithmetic must agree. Thus
`-7 / 3` is `-2` and `-7 % 3` is `-1` in compiled expression lowering. This is
separate from elementwise `floor_divide` and `remainder`, which implement the
floor/divisor-sign operations.

The installed source has a relevant discrepancy: `apply_binop`, used by Python
expression evaluation, uses floor division/remainder, while `BinaryExpr.render`
maps both slash operators to C `/` and comments that shape operands are
nonnegative. These definitions agree on normal nonnegative shape arithmetic.
The signed regression explicitly chooses the compiled C convention; it does
not claim the package's Python evaluator agrees for negative operands.

Mixed integer expressions apply C-style promotion: small integer types promote
to signed 32-bit, unsigned wins an equal-width signed/unsigned tie, and a wider
signed type wins when it represents the narrower unsigned range. Fixtures cover
`i32/u32`, `i64/u64`, and `i64/u32` comparisons. This does not certify all mixed
elementwise operations: `Assign` has its own declared-result-type conversion
rules. The source's integer-literal `LL` spelling heuristic is another distinct
corner and is not covered by these rank tests.

Arithmetic and bitwise expressions also promote two comparison predicates to
signed 32-bit, so `(i < 4) + (i >= 2)` can produce `2` rather than attempting
arithmetic on PTX predicates. Logical `&&` and `||` retain predicate operations.
`test_predicate_promotion.py` checks arithmetic, bitwise operators, shifts,
nonzero division/remainder, and logical operations against integer references.

Integer constants are canonicalized to their declared width and signedness
before folding, including after narrowing. Float-to-integer conversions emit
`cvt.rzi` for both constants and runtime values, preserving PTX's truncation and
destination-range clamping. Large integer-to-FP32 constants and FP64-to-FP32
constants also use `cvt.rn` rather than a host conversion shortcut: rounding a
large integer through Python FP64 can lose which side of an FP32 midpoint it
occupies. For example, `(2**62 + 2**38 + 1)` must round upward to the next FP32
value, while conversion through FP64 first can round downward. See NVIDIA's
[conversion definition](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt).

A typed FP32 literal can itself be spelled as an integer in JSON. Such literals
round their integer significand directly to 24 bits with ties to even before
encoding the PTX bit pattern; they do not first round through Python FP64.
Out-of-range FP32 literals become the appropriately signed infinity.

The cast-edge fixtures compare constant and loaded sources against independent
expectations, including signed/unsigned 64-bit midpoint neighbors and
integer-spelled FP32 literals, FP64
overflow/underflow to FP32, signed zero, and float-to-integer destination limits.
NaN-to-integer checks currently cover 32-bit destinations. These tests specify
the backend's PTX behavior; they do not claim that out-of-range floating casts
have a portable CUDA-C meaning.

## State lifetime and ordered programs

`Loop` and `StridedLoop` initialize each distinct **immediate-body** `Accum`
target before the loop, when `seed=True`. Do not recursively collect nested
accumulators: attention's inner dot product resets on every key, while outer
normalizer and numerator state survive the whole key loop. A nested loop can
shadow a same-named outer accumulator.

`seed=False` updates an existing outer accumulator. Both matmuls explicitly
`Init` their output cells and use `seed=False` on the shared-memory drain loop.
Resetting these cells once per K slab drops all earlier products.

`Accum(name, value, op, base)` means `name = op(base or name, value)`. The seed
still comes from `op.identity`; `base` redirects only the in-loop left operand.
In Emmy 0.3.11 the additive identity is `0`, multiplicative identity `1`, maximum
identity `-1e30`, and minimum identity `1e30`. Use the actual operation metadata
rather than replacing the finite extrema with infinities.

`StridedLoop.end` overrides the axis extent and is evaluated once into the
binding `<axis>_end`; the body may refer to that binding. `unroll` changes only
code generation, not semantics.

An explicit end that is a bare mutable carrier must be copied on loop entry;
otherwise updating that carrier changes the stop condition and `<axis>_end`.
The initial implementation missed this alias: a decreasing bound exited early
and changed the bound observed inside the body. `test_loop_capture.py` retains
the counterexample, including zero/one-trip cases and nested recapture with
retained/unrolled outer loops. The start expression initializes a separate
induction register before the loop, while the step expression is evaluated
after every body execution, matching Emmy's ordinary `for` increment clause.
The regression also changes the step carrier and checks its resulting index
sequence; freezing every loop expression would incorrectly change that case.

Zero-trip seeded loops still expose their identities; zero-trip unseeded loops
preserve their enclosing carriers. An inner fold that becomes empty on a later
outer iteration must reset rather than retain the previous inner result. The
contract-edge fixtures exercise these cases with retained and unrolled loops.

`Assign` and `Let` bindings are lexical; loop-local temporary names do not leak
out, while a loop's accumulated results do. `Cond` preserves mutations of
existing accumulators and keeps newly declared temporaries local. Fresh PTX
registers must not disconnect loop-carried definitions from their consumers.

## Cooperation and asynchronous copies

| Node | Required behavior | Useful regression |
|---|---|---|
| `WarpShuffle` | For XOR offsets `length/2` down to `1`, shuffle **all** state components into `state_b`, then execute `combine_states` in order, updating the original state bindings. | Two-component online softmax; subgroup widths 4/8/16; a partial warp. |
| `TreeHalve` | Reduce pre-populated shared buffers for strides `length/2` down to `1`; every stage ends with a barrier. Finally load every root component into **every thread's** state. | RMSNorm/L2Norm consumer pass, which requires the broadcast. |
| `TreeHalve.inner` | At inner slot `i` and scale `S`, combine `buf[t*S+i]` with `buf[(t+stride)*S+i]`, and broadcast `buf[i]`. | Multiple independent reductions stored as interleaved segments. |
| `Sync` | Preserve block versus warp scope and named barrier ID/count. | Named consumer subset; warp-private shared-memory handoff. |
| `CpAsyncCopy` | Flatten source/destination independently, add `lane_index` when present, and copy exactly 4/8/16 bytes. Use `.ca` for 4/8 and `.cg` for 16. | `m01` has lane offsets and a three-slot ring; `m02` has partial producer participation. |
| `CpAsyncCommit` | Close the current thread's copy group. | More than one prefetched stage. |
| `CpAsyncWait` | Wait until at most `group` groups remain. A following CTA barrier separately makes all producers' completed data available to consumers. | `m01` uses `wait_group 2`, not zero. |

`combine_states` is an executable sequence of `Assign` statements, not an
operation label. Its temporary values and final state writes are ordered. The
softmax combine ends with `copy` into the maximum after all reads of the old
maximum; reducing maximum and denominator independently is incorrect.

Derive the shuffle mask from the IR's complete aligned subgroup. A transient
`activemask` after uneven loop trip counts can miss participants, and an
unconditional full-warp mask can include lanes outside a partial warp. XOR
offsets below the subgroup width preserve the aligned group. All named
participants must reach each shuffle consistently.

`Sync(warp=True)` matches the source's default-mask `__syncwarp()` and emits
`bar.warp.sync 0xffffffff`. It does not infer membership from an instantaneous
active mask after divergent work. Valid input must satisfy the original
full-warp synchronization contract. See NVIDIA's
[warp barrier definition](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync).

Named barriers retain both ID and participant count, including inside a
`TreeHalve`. The contract-edge fixtures synchronize 64 participating threads
inside a 96-thread CTA and reuse a named barrier inside nested loops. A second
barrier orders consumers before the next iteration overwrites shared storage.
All test participants form complete warps and match the specified count.

When `TreeHalve.tid_var == "warp"`, all lanes in a warp have the same tree index.
One representative lane can perform the shared loads/combine/stores, with all
threads still reaching barriers and receiving the final broadcast. This avoids
the reference's redundant same-address stores. Do not apply that lane gate to
arbitrary tree coordinates, where each lane may own a distinct inner slot.

## Shared memory and memory operations

`Smem.extents` are logical element dimensions. Its dtype is a CUDA spelling such
as `float`, unlike scalar statements' `DataType` objects. Honor explicit byte
alignment and at least the element's natural alignment.

Pack buffers by declaration order with `cursor = align_up(cursor, alignment)`.
For launches with dynamic shared memory, use an external shared byte pool and
the packed offsets. `m01` needs 73,728 dynamic bytes: `_a_smem` starts at byte 0
and `_b_smem` at byte 24,576. The other open problems use static declarations
and `smem_bytes=0`; declaring all buffers dynamically would lose their storage.

Vector `Load.names` and `Write.values` are consecutive scalar elements starting
at the given index. Scalar fallback is semantically valid for unsupported
vector widths or insufficiently proven alignment. Atomic writes require one
atomic addition per scalar, rather than a non-atomic vector store. Destination
buffer dtype controls the stored representation.

Alignment facts must respect fixed-width integer wraparound. After runtime
integer addition, subtraction or multiplication, retain only divisors shared
with `2**width`. For example, unsigned 32-bit `1431655766 * 6` wraps to `4`,
then division by `3` yields the odd index `1`. Retaining the factor `6` from
unbounded arithmetic would falsely prove the quotient even and emit a
misaligned vector load. `test_alignment_overflow.py` covers 32/64-bit versions
with both grouped loads and separate scalar loads eligible for packing. The
unsafe baseline is retained as a static address proof; it was not executed.

For software swizzle, the installed definition is
`swizzle(e) = e ^ (((e >> shift) & mask) << 3)` on element indices:

| Mode | Shift | Mask |
|---|---:|---:|
| `NONE` | No permutation | No permutation |
| `B32` | 6 | 1 |
| `B64` | 6 | 3 |
| `B128` | 6 | 7 |

All three plain modes use shift **6**. A suffix such as `B128@7` overrides the
shift only. For `CpAsyncCopy.lane_index=(row,col)`, first add
`row * smem.extents[-1] + col` to the flattened base, then apply the swizzle.
The source's split-swizzle helper is an optional address optimization, not an
additional semantic transform.

## Numerics and scope of evidence

Every open problem uses FP32. The pointwise add and ReLU/multiply cases have
zero tolerance. Preserve their sequential round-to-nearest arithmetic;
FMA contraction is a separate optimization requiring numerical validation.
FP32 literals should be emitted as exact PTX bit-pattern literals. NaN, signed
zero, subnormal, overflow and finite maximum-identity behavior need explicit
tests if coverage claims extend to those inputs.

The open operation set is `add`, `multiply`, `subtract`, `divide`, `reciprocal`,
`sqrt`, `relu`, `maximum`, `exp`, plus `copy` inside state-combination programs.
The scalar operator registry is broader; hidden operation mixes should not be
assumed to stop at this inventory. Unsupported types and operations should
produce precise diagnostics, never silently map to addition or FP32.

Phase 4 adds explicit admission and lowering for FP16/BF16 `BinaryExpr`
arithmetic, including division. Operands expand exactly to FP32 and the result
narrows at each binary-expression node, including inside a wider enclosing
`Let`. `Assign` uses its separate declared-result computation and rounding
policy. Unsupported operation/type pairs fail before PTX is returned; BF16
conversion also depends on the target architecture. See the current
[type rules and coverage limits](TYPE_SUPPORT.md). This expands the tested
narrow-type paths; it does not imply that every operation is legal for every
recognized buffer dtype.

Transcendental lowering is approximate. `tanh.approx.f32` has a documented
maximum relative error of `2**-11`, too loose for some tested tolerances. The
backend instead uses an odd degree-11 Taylor polynomial for `abs(x) <= 0.25`,
and `(1-exp(-2*abs(x)))/(1+exp(-2*abs(x)))` outside that interval. The first path
avoids cancellation near zero; the second avoids positive exponential overflow.
`copysign.f32` restores the input sign, including negative zero. Its PTX operand
order is sign source first, magnitude second. Internal FMAs implement this one
transcendental operation; they do not contract separate IR arithmetic statements.
See NVIDIA's [tanh definition](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh)
and [copysign definition](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#floating-point-instructions-copysign).

For `sin`, `abs(x) < 2**-12` returns the input unchanged: the real sine correction
is below half an FP32 ULP in this interval. This preserves subnormals and signed
zero that approximate argument reduction can otherwise lose. Other sine inputs
and cosine use PTX approximate instructions. The edge suite checks special-value
classes, small/subnormal values and sampled ordinary ranges; it is not a global
correct-rounding proof or validation of arbitrary large-angle reduction.
Likewise, the sampled `tanh` relative-error check does not establish a universal
bound for its composite approximation. See NVIDIA's
[sine definition](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#floating-point-instructions-sin).

The compensated independent-dot policy uses `Dot2`: compute a rounded
product and its FMA residual, apply full `TwoSum` to the high accumulator and
product, add both residuals into a low accumulator, and expose `high + low` at
consumers when `high` is finite. Full `TwoSum` does not require its inputs to be
ordered by magnitude.
This is Algorithm 5.3 in [Ogita, Rump and Oishi, *Accurate Sum and Dot
Product*](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf). It improves accuracy
against a mathematical reference while changing strict sequential FP32
rounding. It is not a proof of correct rounding for every input; overflow and
underflow need separate handling.

Only independent additive dot states qualify. Any in-loop read of the carrier,
including a branch predicate or nested loop bound, disqualifies it. Corrections
belong to carrier-register identities so lexical scope changes cannot attach a
correction to another value. Seeded loops reset both components; `seed=False`
retains both. A later cooperative reduction first materializes the corrected
value into fresh state. An ordinary update consumes its referenced corrected
operands, writes the raw carrier and clears its correction; an explicit base
can replace the old carrier entirely. Pointwise products
and coupled online normalization keep their ordinary lowering.

The nonfinite guard runs at materialization rather than after every product.
`lookup` tests whether the high accumulator is finite, selects the low
correction or positive zero, and emits the original `high + correction` add.
It therefore retains `high + 0` for a nonfinite high instead of replacing the
result with a raw register copy. Inside an independent dot, the high component
changes only by addition: once infinity or NaN occurs, subsequent additions
cannot make it finite. A replacement that can restore a finite value resets
the low component through `seed`, `write_state`, or a fresh cooperative state.
The low component is not separately observable. These facts permit delaying
the guard without changing the finite arithmetic sequence or numerical policy.

`test_compensation_state.py` checks finite cancellation, the product residual,
subnormals, signed zero, infinity/NaN classes and finite-input prefix overflow.
It also covers observation followed by continuation, `Init` and explicit-base
replacement, seeded/empty loops, and a subsequent shuffle. Retained and unrolled
forms compare finite bit patterns and nonfinite classes independently. The
ordinary workload checker still rejects NaNs; special-value acceptance here is
confined to these explicit expected-class tests.

## Source anchors

- [TASK.md](../TASK.md): CLI, ABI, prescribed launch and required/excluded nodes.
- [README.md](../README.md): hidden correctness, code review, then speed.
- Emmy 0.3.11 `compiler/ir/kernel/ir.py`: `Smem` (line 80), `Tile` (183),
  `CpAsyncCopy` (448), `TreeHalve` (743), `WarpShuffle` (834), swizzle map
  (1369), `pack_smem` (2678).
- Emmy 0.3.11 `compiler/ir/stmt/blocks.py`: `Loop` (64), `StridedLoop` (278),
  `Cond` (388).
- Emmy 0.3.11 `compiler/ir/stmt/leaves.py`: `Load` (96), `Assign` (279),
  `Accum` (374), `Init` (581), `Let` (622), `Write` (678).
- Emmy 0.3.11 `compiler/ir/stmt/base.py`: `dtype_promote` (169), ordered
  `render_merge_program` (330), row-major `render_index` (384).
- Emmy 0.3.11 `compiler/ir/elementwise.py`: operation names and identities.

Source rendering methods were read as semantic references only. They must not
be called by the submitted translator.
