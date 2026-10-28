# Launch bounds and scalar edge cases

This revision retains the supplied schedule and launch ABI. It adds one measured
assembler hint and fixes two independently reproduced scalar-lowering gaps.
Public repository research suggested useful experiments; no competitor lowering
code was copied or invoked. The backend still performs its own translation.

## Changes

- **Launch bounds.** The entry declares `.maxntid` from all three `meta.json`
  block dimensions, including auxiliary threads. This tells `ptxas` the actual
  maximum CTA size without changing arithmetic, synchronization or the schedule.
  `compile_file(..., launch_bounds=False)` retains a development control.
- **Loop bindings.** `Loop` and `StridedLoop` induction variables follow Emmy's
  signed 32-bit `int` declaration. An explicit end is another `int` binding,
  evaluated after induction initialization and captured once. Dynamic increments
  compute with their ordinary operand promotions before assignment back to `int`.
- **Floating remainder.** Named FP32/FP64 `remainder`/`mod` no longer form a
  rounded floating quotient. Exact power-of-two scaling and subtraction compute
  the truncating magnitude, followed by divisor-sign adjustment. Subnormals,
  signed zero, infinities and NaNs have explicit handling. FP16/BF16 continue
  through the existing FP32 computation and final narrowing policy.

The remainder algorithm is a correctness fallback. Its worst exponent ratios
need up to 277 loop visits per phase for FP32 and 2,098 for FP64. It is not a
remainder-speed claim. NaN payload/sign is unspecified; floating `BinaryExpr('%')`
remains rejected. Integer remainder is unchanged. The algorithm's subtraction
operands satisfy the exact-subtraction condition; halving only undoes exact
successful doublings, preserving subnormal bits.

The diagnostic comparison harness also uses exact integer equality for
zero-tolerance integer outputs. It preserves integer first-failure values rather
than narrowing them to floating point. The evidence audit rechecks raw runner
exchanges separately from the runner's top-level status.

## Evidence

The final runner gates passed **15/15 public problems and 51/51 additional
packages**. CPU/assembler discovery passed **120 methods**, with 19 explicit
local-GPU skips out of 139 discovered methods and no failures or errors.
No local GPU was used. The source-bound
[release summary](../artifacts/validation/phase8/release-summary.json)
records the completed numerical gates. The [checkout verifier](CHECKOUT.md)
separately checks the file inventory and reproduces the CLI outputs.

| Check | Scope |
|---|---|
| Public runner | 15/15 passed through unchanged `submit.py` |
| Additional runner packages | Final 51/51 passed: 41 retained packages, six loop-binding cases, four floating-remainder cases |
| Remainder inputs | 528 IEEE bit-pattern pairs per type, tested through both named aliases; FP64 uses two I32 words for every input/output, and NaN results are canonicalized |
| CPU/assembler suite | 120 passed; 19 local-GPU methods explicitly skipped; 139 discovered |
| Evidence audit | Lossless integer wire values, ABI, package hashes, raw exchanges and actual CLI PTX |
| Checkout verification | File inventory, current runtime identity, 20 exact CLI reproductions and documentation links |

[Original failures and repaired runs](../artifacts/validation/phase8/counterexamples)
are retained. The old loop start `int64(2**32)` skipped work instead of narrowing
to zero. FP32 `1e8 % 3` and FP64 `1e20 % 3` returned zero instead of one. The
separate unsigned-start/negative-end regression is never launched on the old
compiler because its incorrect comparison can execute billions of iterations.

The first broad run passed 49 of 51 packages. Its two FP64 bit-pattern packages
failed because the runner coerces JSON numbers through binary64, losing low
bits in large I64 payloads. Their original fixtures and failed responses remain
unchanged. The replacement packages transport two I32 words per FP64 value,
assemble/split the bits inside the kernel, and preserve every word through the
observed wire conversion. This fixes the test transport, not the arithmetic
reference or tolerance. The final roster substitutes those two packages; initial
failed attempts are never counted as passes.

The [final feature report](../artifacts/validation/phase8/feature-runner-final/summary.json)
records 51/51 passing checks with unchanged tracked sources. The
[final CPU report](../artifacts/validation/phase8/cpu-regressions-words.json)
includes the revised word-transport tests. A separate CPU audit using exact
rational arithmetic and independent nearest-even IEEE encoding passed all
**1,056 FP64 remainder values** in the two replacement packages. It verified
**4,224 input words and 2,112 output words** through the observed
JSON-number-to-binary64-to-I32 conversion. This audit validates the reference and
transport construction; it is not an additional GPU run. The
[word audit](../artifacts/validation/phase8/final-word-oracle-audit.json),
[retained audit source](../artifacts/validation/phase8/oracle-source) and
[102-exchange evidence replay](../artifacts/validation/phase8/feature-runner-final/integrity-audit.json)
are included for review.

The [new fixtures](../examples/phase8) have independent references. See
[loop tests](../tests/test_loop_bindings.py),
[remainder tests](../tests/test_floating_remainder.py) and
[launch-bound tests](../tests/test_launch_bounds.py).
The existing nine FP64-buffer/FP32-state mathematical-reference conflicts remain
historical failures; this work does not relabel them or claim hidden-set results.

## Performance experiment

We froze the phase-7 compiler and compared it with an independently written
wrapper that adds only `.maxntid`. Each workload had three input checks per
variant, then five alternating-order timing rounds, each with three warmups and
30 timed launches. Outputs were checked after every runner response. All ten
workloads passed. The five-round median for `a02_attention_128` changed from
156.256 to 60.064 microseconds (2.60x); the other five public workloads were
approximately unchanged. Four previously generated nonpublic workloads showed
ratios from approximately 1.00x to 1.05x in this sample.

The [public-workload report](../artifacts/validation/phase8/launch-bounds/paired-runner.json)
and [additional-shape report](../artifacts/validation/phase8/launch-bounds/paired-holdout-runner.json)
retain individual rounds, source hashes, assembly results and correctness checks.
Frozen source copies are beside them. These are comparisons against our own
previous compiler, not competing entries. Runner clocks are not locked, so
small differences are indicative. No universal speedup or ranking is claimed.

The launch-bound experiment was prompted by documented implementations in
[Nightwing](https://github.com/Nightwing-77/emmy/tree/45c673308c91b214fc342da5870d1f14446da43c)
and [StarAtNyte](https://github.com/StarAtNyte/emmy-ptx-compiler/tree/afb5cdaba55bbf2837050313e91337c2fc24e152).
Our implementation follows NVIDIA's
[`.maxntid` contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#performance-tuning-directives-maxntid).
The directive constrains total threads, not individual dimension equality.

## Decisions not adopted

We did not add a compiler backend dependency, copy another entry's translator,
or introduce implicit FP32-to-TF32 arithmetic. A full memory-access verifier,
lane-sharing optimization and additional fragment operations need their own
design and validation. Existing exponential accuracy and sigmoid/SiLU formula
policies remain explicit; a mathematically more stable rearrangement is not
automatically identical to the pinned IR's floating evaluation.

Development and code review were AI-assisted.
