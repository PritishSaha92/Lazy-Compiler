# Scalar type and operation support

The backend emits PTX 8.0 and normally targets `sm_89`. A recognized buffer
dtype is not a promise that every operation is legal for that dtype. Admission
is checked before formatting an instruction; unsupported pairs raise
`CompileError`, with the enclosing statement included in the diagnostic.
`compile.py` prints no partial PTX when compilation fails.

The executable admission table is `_ELEMENTWISE_CAPABILITIES` in
[`ptx_backend.py`](../ptx_backend.py). It records result types and operand counts.
Operand conversions, comparison promotion, bitcast width and target restrictions
are checked separately. This table describes scalar lowering; the separate
[tensor and bulk extensions](FEATURES.md) add their own admission contracts.
The compiler does not implement all NumPy functions or every PTX instruction family.

## Types and node forms

| Type family | Storage and transport | Computation |
|---|---|---|
| FP32 | Scalar and aligned grouped loads/stores | Existing explicit-rounding arithmetic and FP32 composite functions |
| FP64 | Scalar and grouped transport within PTX vector-size limits | Basic arithmetic, comparisons, extrema, FMA, square root, reciprocal and floating integral rounding; transcendental composites rejected |
| FP16 | 16-bit payload in `.b16` registers | Exact expansion to FP32, computation, explicit nearest-even narrowing; comparisons and selections preserve typed values |
| BF16 | 16-bit payload in `.b16` registers | FP32 expansion and explicit narrowing; arithmetic/comparison never requires native `sm_90` BF16 ALU instructions |
| Signed/unsigned integers, 8/16 bits | Memory width is 8/16 bits, virtual register width is 32 bits | Expression integer promotion; explicitly narrow statement results truncate to the declared width; small selections use legal 32-bit instructions |
| Signed/unsigned integers, 32/64 bits | Matching storage width | Fixed-width arithmetic, comparison, bit operations and documented conversions |
| Predicate | Byte memory and `.pred` registers | Logical and comparison results, selection and conversion; arithmetic expressions promote to integer |
| Raw `b32`/`b64` | Bit transport and same-width bitcast | Not generic numeric arithmetic types |

| IR node | Type rule and rounding boundary |
|---|---|
| `BinaryExpr` | Apply the operator's integer promotions. Most arithmetic uses the common operand type; shifts take the promoted **left** operand's type and convert the count separately for PTX. Narrow floating results round at **each binary node**, including a nested expression whose enclosing `Let` has FP32 dtype. Floating `%` and bitwise/shift expressions are rejected. `/` and `//` retain the IR's C-style emitted division spelling; signed integer division truncates toward zero. |
| `Assign` | An explicit result dtype controls the operation. Without one, supported integer arithmetic/bitwise operations use integer rank/sign promotion, with unsigned winning equal-width ties. Other existing defaults are unchanged. A narrow floating operation expands operands to FP32 and narrows the statement result. |
| `FuncCallExpr` | Uses the same operation admission and lowering as `Assign`, with the existing first-operand result-type rule. It is an optional extension beyond the required contest subset. |
| `Accum` and cooperative combine programs | Use the same operation checks; state lifetime, explicit bases and existing FP32 dot policy are unchanged. |
| `Let` / `CastExpr` | Convert the computed expression to the declared dtype with explicit rounding. A final cast does not remove intermediate expression rounding. |
| `TernaryExpr` / `where` | Select already computed typed values. FP16/BF16 payloads are selected with `selp.b16`; small integer selections widen and narrow explicitly. |
| `Builtin` | `thread_idx`, `block_idx`, `block_dim` and `grid_dim` x/y/z components are `u32`; `warp_size` is `s32`. CUDA-spelling aliases have identical lowering. |

The pinned Emmy 0.3.11 `Assign` definition distinguishes declared result dtype
from argument dtype and describes its promote-then-demote path. `BinaryExpr` is
an untyped expression tree whose generated C operator determines expression
promotion. The backend reads these typed classes; it never calls their renderer.
There is a known distinction between Python's negative integer division and the
IR's C rendering; the backend retains its tested C/ PTX truncation rule. Named
`floor_divide`/`remainder` implement floor/sign correction separately.

The [phase-7 release](PHASE7.md) adds mixed-integer inference and shift
counterexamples, including explicit-dtype controls and matching constant/runtime
results. Its validation status is recorded separately from earlier releases.

## Operation families

Aliases (`sum`, `prod`, `sub`, `mul`, `neg`, `true_divide`, `amax`, `max`, `min`,
`absolute`, `pow`, `exp_fast`) normalize before admission.

| Operations | Admitted result types |
|---|---|
| `copy`, `pad`, `where`, comparisons, `logical_not`, `logical_and`, `logical_or` | Numeric types and predicate; conversions remain subject to target restrictions |
| `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `mod`, `remainder`, extrema | Numeric types |
| `negative`, `abs`, `square`, `reciprocal`, `relu` | Numeric types; unsigned negation is modulo-width subtraction from zero, unsigned absolute value is identity |
| `bitwise_and`, `bitwise_or`, `bitwise_not` | Integers and predicates; the historical FP32 logical-mask spelling remains supported |
| `bitwise_xor`, `left_shift`, `right_shift`, `bitwise_count` | Integers; population count counts the absolute value for signed inputs |
| `fma`, `sqrt`, `rsqrt`, `floor`, `ceil`, `trunc`, `rint` | Floating types |
| `exp`, `exp2`, `log`, `log2`, `log10`, `sin`, `cos`, `tanh`, `erf`, `sigmoid`, `silu`, `power` | FP32 and narrow floating results through the existing FP32 composite implementation |
| `bitcast` | Equal-width non-predicate numeric or raw-bit types; 8/16-bit integer and floating register-width differences are handled explicitly |
| Atomic add | FP32, FP64, signed/unsigned 32-bit and unsigned 64-bit; other types are rejected |

`bitwise_count` is a deliberate named-operation semantic correction, not merely
an instruction-admission change. NumPy's named function counts the magnitude of
signed values; the pinned Emmy CUDA/loop render targets instead map directly to
raw population count. The backend follows the named NumPy convention here:
`bitwise_count(i32(-3))` is 2, not 31. For the minimum signed integer, `abs`
retains its high-bit pattern and population count is correctly 1. Separate I16,
I32 and I64 fixtures cover this distinction. The current installed NumPy predates
`np.bitwise_count`, so fixtures use the optional `FuncCallExpr` spelling and an
independent Python `abs(int(value)).bit_count()` oracle; they do not claim that
the local Emmy interpreter evaluated that operation.

This table describes supported lowering paths, not a correctly rounded full
mathematics library. Existing approximate transcendental functions, special-value
policies, and tolerance requirements remain relevant. In particular, FP32
computation followed by narrowing is the declared narrow-composite policy; it is
not advertised as a universally correctly rounded native half-precision FMA or
transcendental implementation. No fast-math, FTZ, reassociation or schedule change
is introduced by admission checks.

## BF16 conversion boundary

PTX 8.0 permits FP32/BF16 conversions on `sm_80+`; direct BF16 conversions to and
from other numeric formats require `sm_90`. The backend uses these rules:

- BF16 to FP32 is exact. BF16 to FP64, FP16, integers or predicate expands to
  FP32 first, then performs the destination conversion. This introduces no
  intermediate loss because every BF16 value is representable in FP32.
- FP16 and 8/16-bit integers to BF16 expand exactly to FP32 and then round once.
- Predicates become FP32 zero/one, then BF16.
- FP32 to BF16 uses `cvt.rn.bf16.f32`.
- FP64 and 32/64-bit integer to BF16 are rejected on `sm_80/89`. On `sm_90+`,
  their native conversion is emitted. A silent FP32 intermediate would introduce
  double rounding for some values and is not an acceptable general fallback.
- Arithmetic conversion involving BF16 is diagnosed below `sm_80`; pure payload
  transport/bitcast need not perform such a conversion.

See NVIDIA's [PTX 8.0 conversion instruction and target notes](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
and [half/BF16 comparison instruction](https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html#half-precision-comparison-instructions-setp).
The BF16 comparison path expands to FP32 because `setp.bf16` requires `sm_90`.
Same-width bitcasts preserve payload bits, including signed zero; they are not
numeric conversions.

## Validation and reproduction

Run CPU and assembler checks without any GPU execution:

```powershell
python tests/test_type_capabilities.py -v
```

The suite tests every admitted same-type operation/result pair through `ptxas`,
conversion and equal-width bitcast matrices on `sm_89` and `sm_90`, selected
unsupported pairs, wrong arity, and actual CLI diagnostic-only failure. Assembly
matrix kernels contain live stores but are **never launched**; numerical fixtures
have valid buffers and separate host references.

Export independently checkable numerical packages for the official runner:

```powershell
python tests/test_type_capabilities.py --export build/type-capabilities/cases
```

Each package contains typed Kernel IR, matching launch metadata, explicit FP32
input data and `expected.npz`. FP32 ABI buffers avoid depending on BF16 support in
the evaluation API while the kernel still computes with explicit FP16/BF16
intermediate types. Cases cover `Assign`, `BinaryExpr`, `FuncCallExpr`, nested
rounding, comparison, conversions, small integer selection, unsigned unary
operations and payload bitcast round trips. These tests separate numerical
lowering from API representation; BF16 storage transport is a separate concern.

Pre-change source and hashes of all fifteen public PTX outputs are retained in
`build/type-capabilities/`. The unchanged public PTX identity check protects the
existing FP32/default kernel behavior. Assembly is evidence of instruction
legality, not GPU numerical execution; organizer-run results belong in the
release validation report with the final backend hash.
