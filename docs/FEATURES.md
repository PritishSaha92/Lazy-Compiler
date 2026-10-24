# Features to inspect during manual review

The public problems are the baseline. This index describes implemented lowering
paths, their examples, and their boundaries. The compiler translates scheduled
typed IR; it does not recognize problem names or replace scalar matmuls with a
different schedule. No CUDA renderer or existing compiler performs its lowering.

| Feature | Implementation | Example and independent tests |
|---|---|---|
| Warp tensor cores | [ptx_tensor.py](../ptx_tensor.py): `mma.sync.aligned.m16n8k16`, FP16 or BF16 multiplicands, FP32 accumulator | [Tensor examples](../examples/tensor/packages), [tests](../tests/test_tensor_cores.py) |
| Tensor fragment transport | Native shared `ldmatrix.x4`, `x2.trans`, paired B loads, transposed B, B32/B64/B128 software swizzles; global gathers with M/N clamping and K zero-fill | Shared, paired, transposed, swizzled and ragged examples in the tensor suite |
| Fragment state and epilogues | Runtime-indexed fragment families, persistent accumulators, guarded stores, scalar bias/ReLU epilogues, transposed output, atomic accumulation and C-to-A repacking | Tensor family, epilogue, atomic and repack tests; original generated ragged FP16 matmul |
| Bulk copy engine | [ptx_bulk.py](../ptx_bulk.py): native `cp.async.bulk`, rectangular pointer-based transfers, transaction barriers and parity waits | [Bulk examples](../examples/bulk), [tests](../tests/test_bulk_copy.py) |
| Combined engine pipeline | Bulk copies feed `ldmatrix` and tensor MMA across three K slabs; accumulator state survives barrier and buffer reuse | [Combined examples](../examples/combined), [tests](../tests/test_feature_interactions.py) |
| Ordinary asynchronous copies | Native 4/8/16-byte `cp.async`, commit groups, nonzero wait groups and separate consumer barriers | [Transport tests](../tests/test_transport_edges.py), [interaction tests](../tests/test_interactions.py) |
| Scalar data types | FP16/BF16/FP32/FP64, signed/unsigned 8/16/32/64-bit integers, predicates and raw-bit transport, with per-operation admission | [Type rules](TYPE_SUPPORT.md), [type tests](../tests/test_type_capabilities.py), [cast tests](../tests/test_cast_edges.py) |
| Required scheduled IR | Tile/raster mapping, scoped loops, explicit accumulator bases, ordered coupled-state reductions, shared broadcast, vector memory and atomics | [Architecture](COMPILER.md), [semantic tests](../tests/test_semantics.py), public problems in all five categories |

## Tensor-core contract

Only the explicit `RegFragment`, `LdmatrixLoad`, `MmaSyncPtx`, `RegStore` and
`FragmentRepack` forms described above are admitted. The supported atom is
**m16n8k16, row-major A / column-major B, FP16 or BF16 inputs, FP32 accumulation**,
requiring SM80 or newer. FP16-accumulator, FP8, block-scaled, Volta and WGMMA
forms are rejected. This extension does not change the numerical policy of
ordinary scalar dot products.

Each lane's fragment registers follow the PTX operand layout. Shared loads use
actual `ldmatrix` instructions. Unstaged operands use typed global loads and
register packing; they are not advertised as shared `ldmatrix` loads. Partial
M/N tiles clamp to an existing row/column, K tails become zero, and fully
overhanging fragments suppress invalid reads. Output guards also enclose
epilogue reads. Shared producers must provide the padding required by staged
loads.

Fragment families contain 1–64 members. Constant invalid indices are diagnosed;
valid runtime indices remain an input-IR precondition. Runtime selection uses
predicated register moves, preserving loop structure instead of forcing large
unrolls. A fragment declaration initializes its FP32 accumulator; placing that
declaration outside a loop preserves accumulated state across iterations.

All participating lanes must execute the same collective instruction. The
compiler checks complete physical warps and compatible Tile partitions. It
does not prove arbitrary branch uniformity. The scheduled IR must satisfy the
PTX collective-participation contract. Shared row alignment is proved from
address divisibility and conservative loop facts; rebinding and repeated-node
cases have rejection regressions.

## Bulk-copy contract

This is **descriptor-free bulk-engine support**, not general host-encoded
`cp.async.bulk.tensor` support. A `TmaDescriptor` provides a static source shape
and box shape. For an in-bounds, unswizzled box, the compiler emits contiguous
row transactions using the original global pointer. The parameter ABI remains
ordinary buffer pointers: no `CUtensorMap` host allocation or encoding is hidden
in the submitted compiler or runner.

The extension supports ranks 1–5, 16-byte-aligned source/destination rows and
row sizes divisible by 16. Source tensors contain fewer than 2^31 elements;
boxes contain fewer than 2^20 bytes and at most 1,024 row segments. Unsupported
swizzles, boundary zero-fill, unknown bounds, integer-overflow risks and invalid
barrier slots are rejected. Bounds come from static dimensions and a small
integer-expression proof domain, never from test inputs.

Initialization publishes the barrier to the asynchronous proxy. The IR retains
its producer/consumer barriers, arrival count, expected transaction byte count,
phase parity and buffer-reuse discipline. These statements must form a valid
protocol; the compiler does not infer a missing arrival or repair a wrong byte
count. A parity wait supplies acquire ordering after bulk completion. The
combined examples verify both data movement and accumulator lifetime over
repeated phases.

Bulk instructions require SM90. The normal CLI selects SM89 for the original
vocabulary and tensor-only kernels, and SM90 when bulk/barrier nodes appear.
An explicitly supplied API target is never silently upgraded. Block and grid
launches remain one-dimensional.

## Additional interaction and architecture checks

The [two-slot ring examples](../examples/ring) exercise five/seven K16 slabs,
both barrier parities, an explicit final drain, FP16 and BF16 computation, and
two CTAs with distinct outputs. BF16 operands are produced by explicit IR
conversion from F32 transport; the runner is not claimed to carry BF16 arguments.
[Ring and compilation-isolation tests](../tests/test_ring_pipeline.py) use
independent exact references and zero tolerance.

Native `CpAsyncCopy`, `CpAsyncCommit` and `CpAsyncWait` require SM80. An explicit
older API target receives a diagnostic before emission, including when nodes
appear in nested control flow. The existing explicit synchronous-copy option
remains available; it is not silently selected. See
[architecture-admission tests](../tests/test_async_target.py).

## Review and reproduction

Start with [SUBMISSION.md](../SUBMISSION.md) and the
[example commands](../examples/README.md). The complete test suite is CPU and
assembler only by default; local GPU execution requires a separate explicit
environment setting. Organizer execution is a separate, serialized command
with retained requests, responses, input hashes and source hashes.

Instruction semantics were checked against NVIDIA's
[PTX MMA specification](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma),
[ldmatrix specification](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix), and
[bulk-copy specification](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk).
Assembly establishes instruction legality; runner comparisons establish the
tested numerical behavior. Neither establishes a hidden-set ranking or a
performance advantage over another submission.
