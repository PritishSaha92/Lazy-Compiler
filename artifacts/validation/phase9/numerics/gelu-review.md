# GELU operation coverage extension

The bounded change implements two existing Emmy operation names, `gelu` and
`gelu_tanh`. It does not add a compiler dependency, delegate lowering, recognize
problem names, change schedules, or alter other mathematical primitives.

## Evidence and contract

- The installed Emmy 0.3.11 source, `emmy/compiler/ir/elementwise.py:61`, defines
  GELU as `0.5*x*(1+erf(x/sqrt(2)))`, and the following line defines the separate
  cubic/tanh approximation. These definitions are the implementation contract.
- [PyTorch's GELU documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html)
  independently documents the distinction between the two formulas. It is
  supporting background, not a replacement for the pinned Emmy contract.
- [PTX floating-point instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions)
  define the arithmetic and conversion instructions used by the existing backend.
- `gelu-baseline.json` records actual compilation failures for six valid typed
  IR packages before the new GELU branch was added. Its hash identifies the
  **pre-GELU working revision**, which may include another agent's independent
  tensor imports; it is not a claim about the immutable `ae302ae` release hash.

## Implementation

The change adds two entries to the existing operation capability table and a
single formula-lowering branch. It reuses the compiler's own `erf` and `tanh`
implementations. Every newly composed arithmetic step is explicit FP32, with no
new contraction or precision change. FP16/BF16 values expand to FP32 and round
once at the enclosing operation boundary, following the existing narrow-type
policy. F64 remains rejected instead of silently using FP32 transcendental math.

The literal formulas retain `GELU(-infinity) = NaN`, `GELU(+infinity) = +infinity`
and the sign of zero. An analytic limiting value at negative infinity is not
substituted for the specified multiplication by zero.

## Independent tests and exported examples

`tests/test_gelu.py` supplies eight packages under `examples/phase9/gelu`:

- Six numeric packages cover both formulas in F32/F16/BF16, with 273 values and
  both `Assign` and `FuncCallExpr` dispatch. The final CTA is partial. Host
  references use `math.erf`/`math.tanh`, explicitly rounded FP32 arithmetic and
  independent IEEE narrow rounding, not the emitter's approximation formulas.
- Two special-value packages cover all three types with 37 raw bit patterns and
  18 exact classification flags per input. They include signed zeros, minimum
  subnormals, normal transitions, maximum finite values, infinities, quiet and
  signalling NaNs, and half-format overflow boundaries. Input payloads and flags
  use I32 buffers, avoiding floating JSON and NaN-payload assumptions.
- Numeric tolerances are `atol=2e-6`, with `rtol=1e-5` for FP32, `1e-3` for FP16,
  and `8e-3` for BF16. These allow the existing approximate transcendental
  primitives and a narrow-format rounding step. Classification cases are exact.
- An additional negative test verifies that both operations still reject F64.

CPU validation completed on 2026-10-28:

```text
python tests/test_gelu.py -v
3 tests passed, including ptxas assembly of all 8 packages for sm_89.

python -m unittest discover -s tests -p test_type_capabilities.py -v
6 tests passed, including the complete admitted operation/type assembly matrix.
```

No GPU launch was performed by this child task. The root agent owns serialized
runner execution and its results must be reported separately. Assembly and host
oracle tests alone do not establish device numerical correctness.

The generated Torch GELU expression packages prepared independently by the root
agent decompose GELU into simpler operations. They are useful integration tests,
but do not replace the direct operation-name fixtures above.
