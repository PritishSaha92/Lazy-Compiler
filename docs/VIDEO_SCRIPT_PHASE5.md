# Read-aloud compiler walkthrough

Read only between the markers. The spoken text has 592 words, approximately 4-5 minutes at a steady technical pace.

<!-- SPOKEN SCRIPT START -->

I'm presenting a Python compiler that translates Emmy's scheduled Kernel IR directly into PTX. I used AI assistance during development. The public problems are my baseline; the optional tensor-core and bulk-copy examples demonstrate the additional instruction families.

The command reads an existing IR JSON file and its neighboring metadata. Emmy supplies typed deserialization. My backend implements instruction selection, addressing, control flow and synchronization. It never compiles CUDA, invokes another compiler backend, or reads expected outputs. NVIDIA's assembler allocates physical registers afterward.

The code has three lowering modules. The core handles scalar expressions and the required scheduled operations. A tensor module handles register fragments and warp matrix instructions. A bulk module handles aligned box copies and transaction barriers. Unsupported combinations fail before any partial PTX reaches standard output.

Tile preserves work ownership. I derive the work-cell index from block and thread IDs, account for auxiliary threads, guard the cell count, and decode the scheduled axes and raster order. StridedLoop uses an induction register, bound check and back edge. It captures its end on entry. The seed flag controls whether immediate accumulators reset or continue existing state; nested reductions reset at their own scope.

WarpShuffle exchanges all coupled state components before executing the ordered combination program. Its participation mask follows the IR subgroup. TreeHalve merges shared-memory partials with the prescribed barriers, then loads the final slot into each consumer. Softmax needs both the coupled merge and the broadcast.

Ordinary asynchronous copies emit cp.async with the original addresses and byte counts. Commit closes a group; wait preserves the requested number of pending groups. Copy completion and consumer synchronization remain separate, so the IR's barriers are retained.

The tensor extension supports the m16n8k16 atom with FP16 or BF16 operands and FP32 accumulation. Shared operands use native ldmatrix. Global operands use per-lane gathers and register packing. M and N tails clamp to valid elements; K tails become zero. Entirely overhanging fragments suppress loads, and output guards enclose epilogue reads. Runtime fragment-array indices use predicated register moves. A fragment declared outside a loop keeps its accumulator across slabs. Complete, uniformly participating warps remain an IR precondition.

The bulk extension uses the SM90 copy engine for aligned, in-bounds rectangular boxes. It lowers descriptor shapes to pointer arithmetic and contiguous row transactions. This is descriptor-free bulk copying, not full host-encoded tensor-map support. Unknown bounds, swizzles and boundary zero-fill are rejected. Initialization publishes the barrier to the asynchronous proxy; expected byte counts and parity waits preserve the transaction protocol. The combined examples copy three slabs, wait, load fragments, accumulate, and synchronize before reusing shared buffers.

Numerics remain explicit. Only structurally eligible independent FP32 dots with nonzero tolerances use compensation. TwoProductFMA recovers the rounded product's residual; TwoSum recovers the addition error. A low component accumulates those errors and follows the same reset rules. Exact or missing tolerances and coupled online normalization keep ordinary lowering. Tensor MMA follows its declared hardware operation instead.

BF16 conversions also respect the emitted architecture. Below SM80, arithmetic conversions are rejected. From SM80, exact FP32 expansion handles BF16 values and safe small sources. FP64 and wide-integer sources need SM90's direct conversion; older targets reject them to avoid silent double rounding. The normal CLI emits SM89, upgrading to SM90 only for explicit bulk or barrier nodes.

Finally, the release includes runnable examples, independent references, negative tests, raw organizer-runner results and source hashes. The public client passes all fifteen problems. The earlier FP64-buffer cases with FP32 state remain documented, including their failed mathematical comparisons. Tests are evidence of implemented behavior, not a guaranteed hidden-set result or an unsupported speed claim.

<!-- SPOKEN SCRIPT END -->

## Screen plan (do not read aloud)

1. Show `compile.py` and the README architecture map.
2. Show `Compiler.tile`, `Compiler.loop`, `shuffle` and `tree`.
3. Show `TensorLowering.global_load`, `shared_load`, `mma` and guarded `store`.
4. Show `BulkLowering._load` and `handles`, then the combined three-slab example.
5. Show [feature limits](FEATURES.md), [release validation](PHASE5.md), and the public-client report.

Use the commands in [examples/README.md](../examples/README.md). The script describes the implementation and discloses AI assistance; reading it does not establish personal understanding. Inspect the referenced functions and examples before recording.

The [previous phase-4 script](VIDEO_SCRIPT_PHASE4.md) is retained as a dated artifact.
