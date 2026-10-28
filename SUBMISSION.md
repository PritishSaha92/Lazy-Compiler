# Submission overview

This Python compiler translates Emmy 0.3.11 scheduled Kernel IR directly into
PTX. It preserves the supplied launch geometry and pointer ABI. Emmy supplies
deserialization and IR definitions; the translation does not invoke its renderer,
nvcc, Triton, LLVM/NVVM or another compiler backend.

**Features beyond the public baseline:** native m16n8k16 FP16/BF16 tensor-core
MMA with FP32 accumulation; shared `ldmatrix` and guarded global fragment loads;
indexed fragment arrays, guarded elementwise fragment operations and ordered row
reductions; direct GELU/tanh-GELU; native SM90 bulk copies with
transaction barriers; and explicit FP16/BF16/FP64/integer scalar type support.
The retained scalar path emits launch bounds from the prescribed block size,
preserves signed integer loop bindings, and computes floating floor remainder
without a rounded quotient. It retains the earlier logical GPU builtin,
auxiliary-thread, mixed-integer and shift-promotion corrections.
The bulk path handles aligned in-bounds boxes through ordinary pointers. It is
not full host-encoded TMA tensor-map support. See the
[precise feature boundaries](docs/FEATURES.md).

## Evidence for this revision

The current compiler passed **15/15 public problems and 67/67 additional
feature packages** (51 retained and 16 new): 82 distinct packages on the
organizers' GPU. CPU/assembler discovery passed **129 of 148 methods**, with 19
explicit local-GPU skips. Source fingerprints stayed unchanged. See the
[release summary](artifacts/validation/phase9/release-summary.json) and
[PHASE9.md](docs/PHASE9.md) for source-bound evidence.

The public client initially completed 14 cases and received a connection reset
on row softmax before any output; its unchanged-source retry passed. Both
reports remain recorded. The feature gate's offline audit rechecked 67 actual
CLI outputs and all 134 retained HTTP exchanges.

The first new-feature run passed all eight direct GELU packages and six fragment
packages. Two other fragment requests were rejected before execution because
their metadata contained nested input lists. Original requests are retained;
flattening those lists preserves all values, shapes, IR and expected results.
They were not counted as device passes. The final 67-package gate subsequently
executed both repaired fixtures successfully.

Separately, the eight generated compositions passed **21/24 mathematical checks**
on both the baseline and final source. The same three FP16 erf-GELU comparisons
remain failures. The [final-output audit](artifacts/validation/phase9/numerics/half-gelu-stage-final.json)
confirms that all 5,439 affected GPU outputs match the declared FP16 stages
bit-for-bit; references and tolerances remain unchanged. These generated
expressions decompose GELU, so they supplement the direct-op fixtures. The
release summary distinguishes the passing feature gate from this failed
mathematical gate; it does not claim every numerical reference passes.

The [phase-8 report](docs/PHASE8.md) retains its completed 15/15 public, 51/51
additional and 120 CPU/assembler passes, with 19 local-GPU skips. It also retains
the original raw-I64 transport failures and corrected two-I32-word fixtures.
Earlier [FP64-buffer/FP32-state conflicts](docs/FINAL_CHECK_2026_10_24.md) remain
reported as nine failed mathematical comparisons. No historical failure is
relabelled as a pass and no hidden-set or performance ranking is claimed.

## Review route

1. Read the [README architecture map](README.md#compiler-architecture).
2. Follow [features → examples → tests](docs/FEATURES.md).
3. Inspect [ptx_tensor.py](ptx_tensor.py) and [ptx_bulk.py](ptx_bulk.py), then their
   negative tests and the exact three-slab [combined tests](tests/test_feature_interactions.py).
4. Run the [example commands](examples/README.md) and inspect the source-bound
   [release evidence](docs/PHASE9.md).
5. Read the [DeepInfer/MoK review and adopted changes](docs/RESEARCH_REVIEW.md).

```console
python -m pip install -r requirements.txt
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
python -m unittest discover -s tests -p test_tensor_cores.py -v
python -m unittest discover -s tests -p test_gelu.py -v
python -m unittest discover -s tests -p test_fragment_rows.py -v
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

Development and code review were AI-assisted. The phase-9 report records the
latest additions, limits and validation status. The competition entry consists of
the public repository and the entrant's project video; the packaging tools do
not publish or submit it.

Release ZIPs and their latest-pointer files are distribution artifacts. Links to
them refer to the repository release layout; they are not dependencies of the
extracted examples or tests. The review archive does not recursively embed itself
or the older runtime ZIPs.
