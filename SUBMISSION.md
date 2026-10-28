# Submission overview

This Python compiler translates Emmy 0.3.11 scheduled Kernel IR directly into
PTX. It preserves the supplied launch geometry and pointer ABI. Emmy supplies
deserialization and IR definitions; the translation does not invoke its renderer,
nvcc, Triton, LLVM/NVVM or another compiler backend.

**Features beyond the public baseline:** native m16n8k16 FP16/BF16 tensor-core
MMA with FP32 accumulation; shared `ldmatrix` and guarded global fragment loads;
indexed fragment arrays and fused scalar epilogues; native SM90 bulk copies with
transaction barriers; and explicit FP16/BF16/FP64/integer scalar type support.
The current revision also emits launch bounds from the prescribed block size,
preserves signed integer loop bindings, and computes floating floor remainder
without a rounded quotient. It retains the earlier logical GPU builtin,
auxiliary-thread, mixed-integer and shift-promotion corrections.
The bulk path handles aligned in-bounds boxes through ordinary pointers. It is
not full host-encoded TMA tensor-map support. See the
[precise feature boundaries](docs/FEATURES.md).

## Evidence for this release

- A fresh run passed **15/15 public problems** through unchanged `submit.py`.
- The final **51/51 additional packages passed**: 41 retained feature packages,
  six new loop-binding packages and four floating-remainder packages.
- CPU/assembler discovery passed **120 of 139 methods**; the other 19 were
  explicitly skipped local-GPU tests. No local GPU was used.
- The checker compares zero-tolerance integer outputs exactly, including mixed
  signed/unsigned 64-bit values. Remainder fixtures transport raw IEEE bits in
  integer buffers, with two I32 words per FP64 value to preserve every bit
  through the runner's wire conversion.
- Paired launch-bound measurements compare this change with our own previous
  compiler. Individual rounds and their correctness checks are retained in the
  phase-8 report; these are not measurements against other submissions.

The [current release summary](artifacts/validation/phase8/release-summary.json)
records the completed gate status and source hashes. The initial feature run's
**49/51 result remains recorded**: two raw-I64 FP64 fixtures lost bits in the
runner's numeric transport. The final roster replaces them with two-I32-word
fixtures, retaining the same reference values and zero tolerance. It does not
relabel those failed attempts. The [phase-8 report](docs/PHASE8.md) explains the
transport audit and final checks. The earlier
[phase-7 results](docs/PHASE7.md), including 41 additional GPU packages and 110
CPU/assembler methods, belong to that earlier source revision. Earlier generated
FP64-buffer/FP32-state reference
conflicts remain documented in the [previous review](docs/FINAL_CHECK_2026_10_24.md);
this release does not relabel those nine failed mathematical comparisons.

## Review route

1. Read the [README architecture map](README.md#compiler-architecture).
2. Follow [features → examples → tests](docs/FEATURES.md).
3. Inspect [ptx_tensor.py](ptx_tensor.py) and [ptx_bulk.py](ptx_bulk.py), then their
   negative tests and the exact three-slab [combined tests](tests/test_feature_interactions.py).
4. Run the [example commands](examples/README.md) and inspect the source-bound
   [release evidence](docs/PHASE8.md).
5. Read the [DeepInfer/MoK review and adopted changes](docs/RESEARCH_REVIEW.md).

```console
python -m pip install -r requirements.txt
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
python -m unittest discover -s tests -p test_tensor_cores.py -v
python -m unittest discover -s tests -p test_bulk_copy.py -v
python -m unittest discover -s tests -p test_feature_interactions.py -v
python -m unittest discover -s tests -p test_ring_pipeline.py -v
python -m unittest discover -s tests -p test_async_target.py -v
python -m unittest discover -s tests -p test_unstamped_integer.py -v
python -m unittest discover -s tests -p test_builtin_aux.py -v
python -m unittest discover -s tests -p test_loop_bindings.py -v
python -m unittest discover -s tests -p test_floating_remainder.py -v
python -m unittest discover -s tests -p test_integer_oracle.py -v
```

Python 3.12+ is required. Keep `meta.json` beside each IR. Compilation requires
no GPU or Torch. `ptxas` enables assembly tests; Torch is a development dependency
for generating workloads and the broader historical reference suite. Full test
discovery is `python -m unittest discover -s tests -v` after installing Torch.
Local-GPU tests are opt-in and remain skipped by default.

This source checkout contains compiler sources, examples, tests, tools,
documentation and selected evidence. `REVIEW_MANIFEST.json` records its files
and hashes. The [checkout notes](docs/CHECKOUT.md) distinguish this inventory
from the preserved phase-7 archive and describe the verification commands.
Use this source checkout for the repository submission; no new ZIP is needed.

Development and code review were AI-assisted. The phase-8 report records the
latest corrections and validation checks. The competition entry consists of
the public repository and the entrant's project video; the packaging tools do
not publish or submit it.

Release ZIPs and their latest-pointer files are distribution artifacts. Links to
them refer to the repository release layout; they are not dependencies of the
extracted examples or tests. The review archive does not recursively embed itself
or the older runtime ZIPs.
