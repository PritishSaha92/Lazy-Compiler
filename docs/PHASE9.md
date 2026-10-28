# Phase 9: activation and fragment-algebra coverage

This revision adds direct `gelu`/`gelu_tanh` operations and bounded
`FragmentApply`/`FragmentRowReduce` lowering. It retains the prescribed schedule,
launch geometry, arithmetic boundaries and existing hardware paths. Compilation
still requires only Python and pinned Emmy 0.3.11; the backend emits PTX itself.
Development and independent source review were AI-assisted.

## Validation status

The current source passed **15/15 public problems and 67/67 additional feature
packages**, 82 distinct packages in total. CPU/assembler discovery passed
**129 of 148 methods**, with 19 explicit local-GPU skips and unchanged source
fingerprints. The [source-bound release summary](../artifacts/validation/phase9/release-summary.json)
collects the numerical, CPU and evidence-integrity gates. Checkout identity is
recorded in its [separate verification report](../artifacts/validation/phase9/checkout/verification.json).
The separate final generated-composition run
passed **21/24 mathematical checks**, with the same three FP16 stage-rounding
conflicts as the baseline and unchanged source fingerprints. The summary's
release acceptance is scoped to the public, feature, CPU and integrity gates;
`all_numerical_references_pass` remains false.

| Completed evidence | Result and scope |
|---|---|
| [Public client](../artifacts/validation/phase9/public-submit/summary.json) and [targeted retry](../artifacts/validation/phase9/public-submit-retry/summary.json) | All 15 unique public problems passed. Initial run: 14 passes and one connection reset before output; unchanged-source row-softmax retry: pass. The initial report remains incomplete. |
| [Final feature gate](../artifacts/validation/phase9/feature-runner-final/summary.json) | 67/67 packages passed: 51 retained and 16 new. Source fingerprints stayed unchanged; offline replay verified 67 CLI outputs and all 134 HTTP exchanges. |
| [CPU/assembler discovery](../artifacts/validation/phase9/cpu-regressions.json) | 129 passed, 19 explicit local-GPU skips, 148 discovered; no failures or errors, sources unchanged. |
| [Initial new-feature run](../artifacts/validation/phase9/new-feature-runner/summary.json) | All eight direct GELU packages and six fragment packages passed device comparison. Two fragment requests were rejected with HTTP 422 before execution because their input `data` was nested rather than flat. |
| [Baseline generated compositions](../artifacts/validation/phase9/compositions-baseline/summary.json) | 21/24 mathematical checks passed; the three FP16 erf-GELU comparisons failed and remain failures. Eight generated packages each used three input sets. |
| [Final generated compositions](../artifacts/validation/phase9/compositions-final/summary.json) | 21/24 mathematical checks passed, with the same three failures and no others; source fingerprints unchanged. |
| [Final half-stage audit](../artifacts/validation/phase9/numerics/half-gelu-stage-final.json) | All 5,439 affected final-run outputs match explicit FP16 IR stages bit-for-bit. The original mathematical gate remains false. The [baseline audit](../artifacts/validation/phase9/numerics/half-gelu-stage-audit.json) is retained separately. |

The two rejected transport requests are neither GPU numerical failures nor
passes. Their [original fixtures](../artifacts/validation/phase9/initial-fixtures/README.md)
and raw exchanges are preserved. The repair only flattens metadata input lists;
values, order, shapes, IR, expected arrays and tolerances remain unchanged. The
fresh 67-package gate executed both repaired packages successfully.
The [transport audit](../artifacts/validation/phase9/tensor/transport-fix.json)
checks the unchanged IR/expected bytes and equality of flattened input values.
No local GPU is used for these checks.

The [final composition integrity audit](../artifacts/validation/phase9/compositions-final/integrity-audit.json)
verifies retained requests, responses and comparisons. Offline audits are not
additional device executions and do not relabel failed mathematical checks.

The [phase-8 release](PHASE8.md) passed 15 public and 51 additional packages with
120 CPU/assembler methods and 19 local-GPU skips. Those completed results belong
to its earlier source hashes. Its performance experiment remains historical;
this revision makes no new performance or competitor-ranking claim. A separate
[compatibility check](../artifacts/validation/phase9/tensor/compatibility.json)
found exact PTX identity for 56 retained modules; this is compilation evidence,
not another GPU run or a new timing measurement. The
[51-package feature identity audit](../artifacts/validation/phase9/retained-ptx-identity.json)
separately verifies the final retained feature inputs and submitted PTX against
phase 8. These overlapping identity checks are not added together as distinct
workloads or used to claim a new speedup.

## Direct activation operations

`Compiler.operation` implements the two separate definitions in pinned Emmy
0.3.11 `emmy/compiler/ir/elementwise.py:61-62`:

```text
gelu(x)      = 0.5*x*(1 + erf(x/sqrt(2)))
gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x*x*x)))
```

The implementation reuses this compiler's existing `erf` and `tanh` primitives.
FP16/BF16 operands expand to FP32 and narrow once at the enclosing operation;
FP64 remains rejected. This is not a new correctly rounded mathematics library.
Literal formula behavior, including negative-infinity NaN and signed zero,
remains explicit. The [type rules](TYPE_SUPPORT.md) state numerical limits.

[Eight direct-operation examples](../examples/phase9/gelu) and
[their independent tests](../tests/test_gelu.py#L51) cover both `Assign` and
`FuncCallExpr`, F32/F16/BF16, partial final CTAs and exceptional classifications.
References use standard-library `erf`/`tanh` and independent narrow rounding.
Special-value inputs use raw I32 words; expected classification flags have zero
tolerance. Ordinary numeric comparisons retain the declared tolerances.

## FP32 C-fragment algebra

[`TensorLowering.apply`](../ptx_tensor.py#L480) and
[`row_reduce`](../ptx_tensor.py#L590) implement the pinned
`FragmentApply`/`FragmentRowReduce` definitions in
`emmy/compiler/ir/kernel/ir.py:993-1226`. Only the modeled **M16N8 layout with
four FP32 C registers per lane** is admitted. This extends the existing
m16n8k16 FP16/BF16 MMA path; it does not introduce another atom or reschedule a
scalar matmul.

`FragmentApply` accepts FRAG elements, ROW scalar pairs, named/expression UNIFORM
values, COORD expressions and GMEM templates. It supports unary post chains,
fresh outputs and in-place updates, including existing indexed fragment
families. Reserved row/column coordinates require explicit origins. A `where`
branch evaluates its selected GMEM operand lazily, so a coordinate mask can
protect an invalid address. Normal scalar operation/type admission still applies.

Mixed-precision ROW, COORD and UNIFORM-expression results of type F64 are
rejected: narrowing them before a binary operation could differ from the pinned
source's final FP32 assignment. Named UNIFORM and GMEM operands follow their
explicit FP32-conversion contract. This diagnostic is tested, rather than
silently treating all mixed expressions as FP32.

`FragmentRowReduce` supports add, multiply, maximum and minimum over nonempty
FP32 C fragments. It preserves the fragment-major left fold, then applies XOR
shuffle masks 2 and 1. Each four-lane group owns the eight columns of two rows;
its results become ROW operands for later fragment operations. Whole physical
warps and Tile partitions are required. Conditional ancestry, dynamic strided
loop controls and enclosing induction-variable rebinding are rejected rather
than assuming collective convergence.

[Eight fragment packages](../examples/phase9/fragments) and
[their tests](../tests/test_fragment_rows.py#L260) cover guarded gathers, post chains,
dynamic families, ordered cancellation, all four folds, MMA normalization and
one complete 16-by-16 attention tile. The last example composes QK, normalization,
explicit FP16 weight repacking and PV. It is not a general online or multi-tile
FlashAttention implementation. Its independent reference includes the same
repacking boundary; small dyadic Q/K/V inputs keep both dot stages exact in FP32.

## Generated compositions and retained numerical conflict

[The eight Torch expressions](../tests/expressions_phase9.py) and
[frozen protocol](../artifacts/validation/phase9/composition-protocol.json) were
defined before runner feedback. They cover tail shapes, FP16, GEMM epilogues,
activation before reduction and activation after attention. The generator
decomposes GELU into simpler operations, so these packages complement the direct
operation fixtures; they do not exercise the new operation names themselves.
The tanh and erf generated IRs retain their distinct formulas despite sharing
the `k_gelu_pointwise` entry name.

The FP16 erf-GELU package explicitly declares five FP16 intermediate values.
Rounding `erf` to half before adding one can lose the negative-tail signal. For
`x=-3.771484375`, the typed stages produce `-0`, while the original mathematical
reference is `-0.00030612945556640625`, outside `atol=0.0002`. An independent
`math.erf` plus IEEE-half-stage oracle matches all 5,439 affected final-run outputs exactly
and reproduces the 9, 8 and 62 mismatches in the three input sets.

The [independent audit and explanation](../artifacts/validation/phase9/numerics/half-gelu-stage-review.md)
retain per-stage values and the original input/response identities. The generated
CUDA reference inlines away some intermediate half assignments,
which is a different arithmetic path. This compiler preserves the declared IR
boundaries. The mathematical reference and tolerance are unchanged, the three
failures remain visible, and no workload-name exception or global widening was
added. The older nine FP64-buffer/FP32-state conflicts also remain documented in
[the 24 October review](FINAL_CHECK_2026_10_24.md).

## Review and reproduction

```console
python -m unittest discover -s tests -p test_gelu.py -v
python -m unittest discover -s tests -p test_fragment_rows.py -v
python compile.py examples/phase9/fragments/fragment_attention_tile_f16/k_regression_fragment_attention_tile_f16.ir.json > attention.ptx
python tools/check_feature_examples.py examples/phase9/gelu/activation_gelu_f32 examples/phase9/fragments/fragment_family_row_softmax --report-dir build/my-phase9-check
```

The default unit tests use CPU references and `ptxas`; the final command uses the
organizers' runner. Keep each IR beside its `meta.json` and use a new report
directory. Source identities, full requests and responses distinguish assembly,
transport, numerical results and report acceptance.

A follow-up public-repository survey found no additional confirmed competing
backend across twelve repository queries plus code/commit searches. Sixteen
default HEADs and four implementation branches were unchanged from the earlier
survey. This was a bounded source review, not an exhaustive inventory or an
execution-based comparison. The additions were implemented from pinned IR
semantics with independent tests; competitor implementations were not executed
or copied.
