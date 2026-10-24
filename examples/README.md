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

Run the focused CPU/assembler tests:

```console
python -m unittest discover -s tests -p test_tensor_cores.py -v
python -m unittest discover -s tests -p test_bulk_copy.py -v
python -m unittest discover -s tests -p test_feature_interactions.py -v
python -m unittest discover -s tests -p test_ring_pipeline.py -v
python -m unittest discover -s tests -p test_async_target.py -v
```

If `ptxas` is unavailable, assembler-specific methods report a skip. These
commands do not launch a local GPU. For an organizer-GPU execution with raw
wire evidence, choose a new report directory:

```console
python tools/check_feature_examples.py examples/combined/bulk_tensor_k48_f16 examples/combined/bulk_tensor_k48_bias_relu_transposed --report-dir build/my-feature-check
python submit.py p00_scalar_add --compiler compile.py
```

The typed-example definitions and independent references are in
[test_tensor_cores.py](../tests/test_tensor_cores.py),
[test_bulk_copy.py](../tests/test_bulk_copy.py) and
[test_feature_interactions.py](../tests/test_feature_interactions.py).
Their `--export NEW_DIRECTORY` commands regenerate packages. The generated
matmul retains its original IR, metadata and reference bytes; its provenance
is recorded beside the copy.

The current 29-package corpus is pinned by [research-feature-manifest.json](research-feature-manifest.json). The previous 26-package manifest remains historical. Ring fixtures can be regenerated with `python tests/test_ring_pipeline.py --export NEW_DIRECTORY`.
