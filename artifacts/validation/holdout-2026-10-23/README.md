# Held-out workload split: 23 October 2026

This corpus contains 18 original synthetic workloads across the competition's
five public categories. It uses no organizer hidden materials. Its purpose is
to evaluate a compiler policy selected using the earlier public/development
sets. Successful IR generation and CPU reference checks do not establish PTX
correctness or performance.

The expression source and evaluation conditions were recorded before generation
in `protocol-source-freeze.json`. Disjoint-axis, adjacent-axis, and global 3D
mean proposals failed inside Emmy 0.3.11; the final replacement is an odd-width
2D row mean-square reduction. Two fused
proposals needed their intermediate expressions inlined: Emmy traces the final
expression and otherwise treats a setup intermediate as a new input. All
generation repairs and the source-hash chain are disclosed in
`generation-repairs.json`. The earlier attempts remain available. No workload or tolerance was
changed in response to a PTX result: the compiler has not been evaluated on this
split at freeze time.

`manifest.json` records the final artifact freeze, source versions, baseline
compiler fingerprints, every package file hash, and category counts.
`MANIFEST.sha256` seals that manifest. Verify before and after evaluation with:

```console
python tests/freeze_holdout.py --verify
```

## Workloads

| Category | Workloads | Structural differences from the earlier ten fresh cases |
|---|---:|---|
| Pointwise | 3 | Row/column broadcasting, a 3D two-branch expression, odd-width rational sigmoid gating |
| Reductions | 4 | Column sum/max, odd-width row mean square, two-input row inner product |
| Matmul | 4 | 24x1536x40 long-K, 192x40x12 tall-thin, 12x48x224 wide-output, bias/ReLU epilogue |
| Fused | 4 | Mean-centered affine normalization, residual L2 plus weights, weighted softmax reduction, RMS sigmoid gate |
| Attention | 3 | Q/KV lengths 8/192 and 160/24, plus a 12/80 attention with residual epilogue |

The attention inputs contain pretransposed K, as required by the current Emmy
frontend. There are no causal masks, dynamic shapes, or tensor-core operations
requested by these expressions. Generated statements remain authoritative;
manifest creation records files without asking our backend to interpret them.

## Frozen evaluation conditions

Each workload has three planned correctness conditions: normal random inputs at
seed offsets 37 and 89, and normal inputs scaled by eight at offset 137. This is
54 planned cases. Tolerances are fixed in the expression source and metadata.
Exact pointwise and maximum cases retain zero tolerance.

The freeze check also verifies traced input names, order, and shapes against the
original random-input declarations, catching accidental setup-intermediate
substitution. NumPy references were checked against ordinary CPU PyTorch using those same
input conditions, independently of Emmy and the PTX compiler. The detailed
54-case record is `reference-sanity.json`. FP64 intermediate arithmetic supplies
reduction, normalization, and matrix-product references; this is an oracle, not
a permitted replacement for the submitted compiler.

Freeze the candidate compiler/policy fingerprint before the first PTX
evaluation. Keep baseline and candidate outcomes, including failures. If a
result from this split causes a policy change, disclose this split as
development data and use another untouched split for a new generalization
claim. Do not remove failing workloads or relax their tolerances.

Performance evaluation must follow correctness, use identical launch metadata,
and serialize GPU measurements. Compare interleaved baseline/candidate rounds
and retain variability and compiler/code-size costs. This manifest does not
authorize or claim a performance run; the root workflow coordinates that gate.

## Provenance and reproduction

The source is `tests/expressions_holdout.py`. Emmy 0.3.11 creates scheduled Kernel
IR and matching metadata through the repository's generation tooling. Its CUDA
renderer is used only to construct those test packages. The participant
compiler still translates Kernel IR directly into PTX.

Generate into a **new** directory; preserve this frozen directory:

```console
python tests/generate_stress.py --expressions tests/expressions_holdout.py --out artifacts/validation/holdout-regenerated --target sm_89
```

The Windows generation wrapper uses an import-only `fcntl` shim, disabled
golden inventories, and isolated temporary tuning state. An attempted lock
raises an error. It does not modify installed Emmy or consult machine-local
performance evidence.
