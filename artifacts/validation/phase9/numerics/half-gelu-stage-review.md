# Recorded FP16 GELU composition failures

The three recorded baseline failures are explained by the generated Kernel IR's
FP16 stage boundaries. They do not provide evidence that the emitter's `erf`
approximation produced an incorrect half result for these inputs.

`audit_half_gelu_stages.py` reads the saved requests and responses only. It uses
`math.erf`, float64 host intermediates and NumPy IEEE-F16 conversion to model
each typed assignment independently of the emitter's polynomial. No compiler
code is imported, no GPU is launched, and no reference or tolerance is changed.
Results and input/response hashes are in `half-gelu-stage-audit.json`.

| Input | Elements | GPU versus typed-stage half bits | Original mathematical-reference failures |
|---|---:|---:|---:|
| seed 0, normal | 1,813 | 0 mismatches | 9 |
| seed 17, normal | 1,813 | 0 mismatches | 8 |
| seed 101, wide | 1,813 | 0 mismatches | 62 |

All 5,439 GPU results match the independent stage oracle bit-for-bit, including
zero signs. The stage oracle reproduces all 79 original tolerance failures.
The seed-zero mathematical oracle also matches the frozen expected file exactly.

The generated IR explicitly declares five FP16 values:

```text
v0 = half(half(inv_sqrt2) * x)
v1 = half(erf(v0))
v2 = half(1 + v1)
v3 = half(0.5 * x)
v4 = half(v2 * v3)
```

The source Torch GELU reference instead computes the full mathematical formula
and rounds its output to half. For `x = -3.771484375`, the staged `erf` rounds to
`-1`, so the typed program outputs `-0`; the mathematical reference is
`-0.00030612945556640625`, outside the frozen `atol=0.0002`. A more accurate
FP32 `erf` implementation cannot recover the discarded half-stage bits.

The generated `.ref.cu` introduces another distinction: its expression inlining
eliminates the intermediate half declarations for `erf` and the addition. It
retains half input products but performs the `erff` result's mixed arithmetic in
float until final assignment. A host model of this inline expression has zero
mathematical-tolerance failures on these three input sets. That observation is
a diagnostic model, not a compiled or executed CUDA-reference result.

Recommendation: preserve these failed mathematical integration cases and report
the contract conflict. Do not widen all typed half statements or special-case
the workload name to make them pass. The new direct `gelu` operation has one
operation boundary and therefore can correctly compute in FP32 before one half
conversion; this does not authorize erasing separate boundaries in an arbitrary
already decomposed program. An explicitly promoted Torch source would be a new
test, not a replacement for the failed case.
