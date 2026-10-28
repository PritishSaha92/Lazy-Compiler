# Compiler examples

These packages contain input IR, adjacent launch metadata and independent
expected arrays. They are development examples, not precompiled answers used
by the compiler. The compiler reads only IR and launch metadata.

Run from the repository or extracted review-bundle root with Python 3.12+:

```console
python -m pip install -r requirements.txt
python compile.py examples/tensor/packages/tensor_global_f16/k_regression_tensor_global_f16.ir.json > tensor.ptx
python compile.py examples/bulk/bulk_rectangle/k_regression_bulk_rectangle.ir.json > bulk.ptx
python compile.py examples/combined/bulk_tensor_k48_f16/k_regression_bulk_tensor_k48_f16.ir.json > combined.ptx
python compile.py examples/ring/ring_tensor_k80_f16/k_regression_ring_tensor_k80_f16.ir.json > ring.ptx
python compile.py examples/phase7/integer/unstamped_mixed_integer_add/k_regression_unstamped_mixed_integer_add.ir.json > integer.ptx
python compile.py examples/phase7/builtins/auxiliary_raw_thread_named_barrier/k_regression_auxiliary_raw_thread_named_barrier.ir.json > auxiliary.ptx
```

PowerShell 5.1 users should wrap the emission command in `cmd /c "... > out.ptx"`
to avoid UTF-16 redirection. Compilation needs no GPU, CUDA toolkit or Torch.
Do not move an IR away from its companion `meta.json`.

| Directory | What it demonstrates |
|---|---|
| [tensor/packages](tensor/packages) | FP16/BF16 tensor instructions, shared/global operand transport, tails, fragment families and epilogues |
| [bulk](bulk) | Aligned linear/rectangular/rank-five bulk transfers, barrier phases, ring slots and exact FP64 payloads |
| [combined](combined) | Bulk-copy → shared `ldmatrix` → tensor-core MMA over three K slabs; fused epilogue and transposed output |
| [ring](ring) | Two-slot pipelined copies and MMA across five/seven slabs, repeated barrier phases, final drain and disjoint CTAs |
| [generated_tensor](generated_tensor) | The original Emmy-generated 11×7 by 7×19 FP16 matmul, previously outside the scalar backend's vocabulary |
| [../problems](../problems) | The 15 public baseline problems, including online softmax and attention-like state |
| [phase7/integer](phase7/integer) | Seven exact-reference packages: mixed-integer operations, constant/runtime shift promotion and explicit-dtype controls |
| [phase7/builtins](phase7/builtins) | Four packages: 3-D raw builtin values, unsigned geometry semantics, explicit 64-compute/32-helper roles, and multi-CTA bulk coordinates |
| [phase7/retained](phase7/retained) | One unchanged scalar package rechecked because its emitted shift-count type changed |
| [phase9/gelu](phase9/gelu) | Eight direct GELU/tanh-GELU packages: F32/F16/BF16 and exact exceptional classifications |
| [phase9/fragments](phase9/fragments) | Eight fragment-algebra packages: guarded GMEM, ROW broadcasts, ordered folds, MMA normalization and one complete attention tile |
| [phase9/generated](phase9/generated) | Eight original Torch compositions; retained FP16 stage-rounding conflict is documented in the phase-9 report |
| [phase8](phase8) | Six loop-binding packages and four current floating-remainder packages; original failed I64 transport fixtures are retained separately from the final roster |

Run the focused CPU/assembler tests:

```console
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
python -m unittest discover -s tests -p test_launch_bounds.py -v
```

If `ptxas` is unavailable, assembler-specific methods report a skip. These
commands do not launch a local GPU. For an organizer-GPU execution with raw
wire evidence, choose a new report directory:

```console
python tools/check_feature_examples.py examples/combined/bulk_tensor_k48_f16 examples/combined/bulk_tensor_k48_bias_relu_transposed --report-dir build/my-feature-check
python tools/check_feature_examples.py examples/phase7/builtins/auxiliary_raw_thread_named_barrier examples/phase7/integer/shift_signed_right_unsigned_count --report-dir build/my-phase7-check
python tools/check_feature_examples.py examples/phase8/loops/strided_i64_start_narrows examples/phase8/remainder_words/floating_f64_mod_words --report-dir build/my-phase8-check
python submit.py p00_scalar_add --compiler compile.py
```

The typed-example definitions and independent references are in
[test_tensor_cores.py](../tests/test_tensor_cores.py),
[test_bulk_copy.py](../tests/test_bulk_copy.py) and
[test_feature_interactions.py](../tests/test_feature_interactions.py).
Their `--export NEW_DIRECTORY` commands regenerate packages. The generated
matmul retains its original IR, metadata and reference bytes; its provenance
is recorded beside the copy.

The previous 29-package corpus is pinned by
[research-feature-manifest.json](research-feature-manifest.json). The earlier
[phase-7 manifest](../artifacts/validation/phase7/new-feature-manifest.json)
adds 11 packages; the [retained scalar
manifest](../artifacts/validation/phase7/changed-scalar-manifest.json) records
the twelfth package under `phase7`. All 41 additional packages and all 15 public
problems passed organizer-GPU checks for phase 7; see
[PHASE7.md](../docs/PHASE7.md). The current
[phase-8 roster](../artifacts/validation/phase8/feature-manifest-final.json)
adds ten packages. All 51 additional and 15 public packages passed with the
phase-8 compiler; see [PHASE8.md](../docs/PHASE8.md). The phase-9 final roster
contains those 51 plus 16 new typed packages; all **67/67** passed alongside
**15/15 public** problems. [PHASE9.md](../docs/PHASE9.md) records the current
results, preserved transport attempts and separate mathematical-reference
failures in generated compositions. These are development correctness
tests, not hidden-set results.

Phase-9 definitions and independent references are in
[test_gelu.py](../tests/test_gelu.py) and
[test_fragment_rows.py](../tests/test_fragment_rows.py); each supports
`--export NEW_DIRECTORY`. Generated integration expressions are in
[expressions_phase9.py](../tests/expressions_phase9.py). They decompose GELU into
primitive nodes, so they do not replace the direct-operation examples.

Ring fixtures can be regenerated with
`python tests/test_ring_pipeline.py --export NEW_DIRECTORY`. The new definitions
and independent references are in [test_unstamped_integer.py](../tests/test_unstamped_integer.py)
and [test_builtin_aux.py](../tests/test_builtin_aux.py); each supports
`--export NEW_DIRECTORY`. The 3-D builtin example does not use `Tile`; general
3-D Tile schedules and `SetMaxNReg` remain outside the supported contract.
