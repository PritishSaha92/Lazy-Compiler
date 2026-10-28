# Submission overview

This Python compiler translates Emmy 0.3.11 scheduled Kernel IR directly into
PTX. It preserves the supplied launch geometry and pointer ABI. Emmy supplies
deserialization and IR definitions; the translation does not invoke its renderer,
nvcc, Triton, LLVM/NVVM or another compiler backend.

**Features beyond the public baseline:** native m16n8k16 FP16/BF16 tensor-core
MMA with FP32 accumulation; shared `ldmatrix` and guarded global fragment loads;
indexed fragment arrays and fused scalar epilogues; native SM90 bulk copies with
transaction barriers; and explicit FP16/BF16/FP64/integer scalar type support.
The latest corrections cover logical GPU builtins with unsigned geometry
semantics, explicit auxiliary-thread role guards, mixed-integer inference and
shift promotion.
The bulk path handles aligned in-bounds boxes through ordinary pointers. It is
not full host-encoded TMA tensor-map support. See the
[precise feature boundaries](docs/FEATURES.md).

## Evidence for this release

- **15/15** public problems passed through the unchanged organizer `submit.py`.
- **41/41** additional packages passed on the organizer GPU: 29 previous feature
  examples, 11 new integer/builtin/auxiliary counterexamples and one retained
  scalar case whose PTX changed. The previous examples cover tensor cores,
  bulk copies, combined pipelines, repeated ring reuse and generated ragged FP16.
- **110 CPU/assembler test methods passed**; 19 local-GPU methods were explicitly
  skipped. No local GPU was used for this release.
- The isolated install audit passed 25 packages, with 75 successful CLI/audited
  compilations and three expected diagnostics.
- PTX remained byte-identical for 119 of 120 retained scalar IR inputs and all
  29 previous optional-feature modules. The one changed scalar module passed
  a fresh organizer-GPU check.

The [release report](docs/PHASE7.md) binds these results to source hashes and
retains raw feature-runner exchanges. These are correctness results, not a
measured speed ranking. Earlier generated FP64-buffer/FP32-state reference
conflicts remain documented in the [previous review](docs/FINAL_CHECK_2026_10_24.md);
this release does not relabel those nine failed mathematical comparisons.

## Review route

1. Read the [README architecture map](README.md#compiler-architecture).
2. Follow [features → examples → tests](docs/FEATURES.md).
3. Inspect [ptx_tensor.py](ptx_tensor.py) and [ptx_bulk.py](ptx_bulk.py), then their
   negative tests and the exact three-slab [combined tests](tests/test_feature_interactions.py).
4. Run the [example commands](examples/README.md) and inspect the source-bound
   [release evidence](docs/PHASE7.md).
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
```

Python 3.12+ is required. Keep `meta.json` beside each IR. Compilation requires
no GPU or Torch. `ptxas` enables assembly tests; Torch is a development dependency
for generating workloads and the broader historical reference suite. Full test
discovery is `python -m unittest discover -s tests -v` after installing Torch.
Local-GPU tests are opt-in and remain skipped by default.

This source checkout contains compiler sources, examples, tests, tools,
documentation and selected evidence. `REVIEW_MANIFEST.json` records its files
and hashes. The [checkout notes](docs/CHECKOUT.md) distinguish this inventory
from the preserved release archive and describe the verification commands.
The smaller runtime ZIP contains only the five runtime files.

Development and code review were AI-assisted. The phase-7 report records the
latest corrections and validation checks. The competition entry consists of
the public repository and the entrant's project video; the packaging tools do
not publish or submit it.

Release ZIPs and their latest-pointer files are distribution artifacts. Links to
them refer to the repository release layout; they are not dependencies of the
extracted examples or tests. The review archive does not recursively embed itself
or the older runtime ZIPs.
