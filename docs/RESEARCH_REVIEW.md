# DeepInfer and Mixture of Kittens: review and decisions

The review led to one small compiler correction and stronger interaction tests.
It did not justify replacing the scheduled-IR backend with a distributed runtime,
changing the launch geometry, or introducing lower-precision arithmetic.

## Source record

DeepInfer was reviewed at commit
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`. The local checkout contains 336
UTF-8 text files, totaling 63,630 lines, and six images. All text files were read
in full and all six images inspected; the merged record verifies their hashes
against the original inventory. Its six vendor submodule
directories contain no files in this checkout; no dependencies or model weights
were installed. The [file-level review record](research/source-review-manifest.json)
distinguishes the source inventory from completed reading and image inspection.

The [Mixture of Kittens article](https://cursor.com/blog/mixture-of-kittens) was
read through its unique article body, including the algorithms, tables,
implementation discussion, evaluation and footnote. The page repeats content
for different display layouts. Its HTML and extracted text are retained in the
workspace's private research folder, outside this submission. The review also
used the [project's own requirements](https://github.com/cursor/mixture-of-kittens#setup)
and [NVIDIA's instruction specifications](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html).
No third-party runtime or kernel implementation is included in our compiler.
All six distinct technical article figures were downloaded and inspected,
including the scheduling figure that initially failed through the web reader.
Light/dark duplicates and author portraits are not additional technical figures.

## What DeepInfer contributes

The source makes several useful contracts explicit:

- **Rounding belongs to the operation.** The residual normalization path rounds
  the residual addition in the input dtype before the FP32 reduction. Fused
  transforms also retain a particular accumulation order. This supports keeping
  our typed expression boundaries and ordered reduction programs. See
  [normalization](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/norm.py)
  and [fused transforms](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_transform.py).
- **Arrival and reuse are separate events.** Its shared-memory broadcast ring
  distinguishes unread, partly consumed and reusable slots. Its worker retains
  transfer sources until their asynchronous consumers are finished. These are
  lifetime principles; host flags are not substitutes for PTX memory ordering.
  See [the ring protocol](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/distributed/shm_broadcast.py).
- **Test isolation is observable.** Batch-invariance tests compare identical work
  in different surrounding contexts. Cache tests use wrapped positions, poison
  values and reset cached class state. Our new compilation-order regression
  applies this principle to compiler state rather than claiming batch-invariant
  GPU execution. See [batch invariance](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/tests/models/test_batch_invariant.py).
- **A benchmark needs a complete configuration.** DeepInfer's benchmark keeps
  raw timing/token windows and effective settings, separates warmup, and labels
  load-balancing contamination. Its sweep refuses mixed comparison settings.
  Our single-iteration runner checks remain correctness evidence and indicative
  timing, not a performance ranking. See
  [benchmark.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/benchmark.py).
- **A storage dtype is not an instruction contract.** Packed quantized weights
  require their own scale layout, block shape and hardware assumptions. Adding
  an FP4 spelling would not implement block-scaled MMA. The existing explicit
  type/layout/target admission rules remain the right boundary.

Not every testing choice transfers. DeepInfer's `testing.py` helper replaces
infinities and uses cosine difference as an acceptance criterion. Our compiler
checks retain the prescribed componentwise tolerances, output shape and dtype,
and special-value behavior. Serving-only approximation, such as bounded replay,
is not a legal default transformation of arbitrary scheduled IR.

The Ascend `rstd_1ulp.py` path also explicitly uses TwoSum/FMA residuals, retains
operation order and restricts the epsilon domain. Its quantization code handles
double rounding with representation-specific logic. These are useful independent
examples of precise numerical contracts. Our existing compensated-dot policy
predates this review and is not broadened by it.

## What Mixture of Kittens contributes

MoK combines distributed MoE computation and communication, uses reusable ring
storage, and controls scheduling and floating-point order. Its overlap depends
on workload granularity and hardware resources. The transferable lesson is to
test buffer ownership and completion across repeated reuse, and measure complete
configurations. Its NVL72 throughput results do not establish a benefit for this
single-kernel contest backend. [Article](https://cursor.com/blog/mixture-of-kittens)

The project targets SM100/SM103 with CUDA 13, while this compiler emits PTX 8.0
for the prescribed single-device ABI. A GPU's architecture family name alone
does not establish instruction compatibility. CLC is not categorically absent
from SM120, but adopting it would require newer PTX and a work-cancellation
protocol absent from our lowering contract. Neither TMEM nor distributed
dispatch is added on the strength of another workload's performance.

## Implemented changes

### Native asynchronous-copy target admission

An explicit pre-SM80 API target previously produced native `cp.async` copy,
commit and wait instructions that the assembler rejected. The constructor now
diagnoses those nodes, including nested bodies, before emission. The existing
explicit `synchronous_copy=True` implementation remains available. Supported
SM80/SM89 native emission is unchanged. This follows the
[PTX 8.0 target requirements](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).

The [six regression methods](../tests/test_async_target.py) cover each native
node, nested control flow, a scalar older-target kernel, legal targets and the
explicit fallback. The retained pre-fix example records the original assembler
errors; it is historical failure evidence, not a passing release gate.

### Combined ring-pipeline tests

The [new fixtures](../tests/test_ring_pipeline.py) prime slot zero, start the
next transfer before consuming the current slot, and drain the final slab
separately. For slab `s`, the slot is `s % 2` and the wait parity is
`(s // 2) % 2`. The consumer waits for its copies, loads fragments, performs MMA,
and reaches a CTA barrier before that storage can be overwritten. The
accumulator is declared outside the loop.

Five- and seven-slab cases cover repeated wraparound, both barrier parities and
the final drain. One case uses two CTAs with distinct output rows. The BF16
case transports F32 input buffers and explicitly converts shared values to BF16;
it does not claim BF16 support in the organizer's argument transport.

All inputs are small, exactly representable dyadic values. Independent FP64
matrix calculations establish references exactly representable in FP32. Zero
tolerance therefore exposes missing, stale or mismatched slabs without relying
on numerical slack. CPU checks establish that those mutations change the
reference and that the CTAs have different results.

### Compilation isolation

The same-process test alternates strict scalar compilation, tensor/bulk kernels,
compensated dots and widened dots, then recompiles earlier inputs. It checks
identical PTX and unchanged input files. This exercises target, register, label,
buffer, fragment and numerical-mode isolation.

Deterministic PTX generation is separate from deterministic GPU execution. An
IR using split-K floating atomic additions can retain scheduling-dependent
rounding; the compiler does not turn that operation into a fixed-order reduction.

## Pipeline decisions

The required sequence remains: preserve typed semantics, reject unsupported
combinations, validate independent references, then measure legal alternatives.
No tuning candidate is promoted from one favorable timing. Any later comparison
must hold inputs, target, numerical settings and launch metadata fixed, retain
failures, separate profiling from timing, and evaluate held-out workload families.

No new RL policy, persistent scheduler, NVLink transport, quantization mode,
programmatic-dependent-launch protocol or barrier-removal optimization is added.
Each would need a suitable IR contract and evidence of benefit. See the current
[release validation](PHASE6.md) and [feature boundaries](FEATURES.md).
