# Deferring the compensated-dot finite guard

This experiment moves the guard on a Dot2 correction from every implicit
accumulation to each observation of that compensated carrier. It retains the
same rounded product, FMA product residual, full TwoSum and correction addition
in the same iteration order. It changes neither the reduction order nor the
precision policy. Production promotion depends on the separate paired timing
experiment and complete validation gate.

Frozen sources are `guard-baseline/compile.py` and
`guard-candidate/compile.py`. Baseline backend SHA-256 is
`6adcba46aef59de0eb41555ce19de22e87f1980fbcdc8fff2a24830c5be44ec9`;
candidate backend SHA-256 is
`8afa55d5ac32da2eda6184058de9aa40ee8327366fc9fa6261b82fb0ed2ce002`.
These hashes describe actual file bytes, including Windows newlines.

## Observable-state proof

Let `h` be the high carrier and `l` its correction. The original update ends
with `h = RN(h + p)` and selects correction zero when this new high value is
nonfinite. The candidate retains the same high update and saves the calculated
correction unconditionally. It materializes an observed value as

```text
RN(h + (isfinite(h) ? l : +0))
```

This still performs the original `h + 0` operation when high is nonfinite; it
does not substitute a direct return of `h` with different NaN arithmetic.

The invariant between explicit state resets is:

1. Both versions have exactly the same high carrier.
2. Whenever high is finite, both versions have exactly the same correction.
3. Whenever high is nonfinite, the old correction is zero and the new
   correction is unobserved. Both materialize the same `RN(h + 0)` result.

An implicit high update uses only its old high and the rounded product, never
the correction. IEEE addition cannot turn an infinity or NaN into a finite
value: adding finite values retains infinity, opposite infinities make NaN,
and NaN stays NaN. Consequently an invalid deferred correction cannot later
become visible through a newly finite high during the same accumulation.

Every operation that can replace high with a new finite value establishes a
new state: `seed` and `Init` create a zero correction; `write_state` consumes an
already materialized value and clears correction; `prepare_state` materializes
the old value and initializes its replacement correction to zero. Shuffle
combines and final tree replacements use those same paths. No raw correction
is an IR value, and all observations go through `lookup`.

Thus finite arithmetic and all observable carrier values are preserved. For
nonfinite high states, the raw hidden correction may differ, but cannot affect
an observation or a later reset. This argument does not assert that Dot2
repairs an overflowing FP32 high prefix into a finite exact dot product.

## Independent numerical coverage

`tests/test_compensation_state.py` constructs seven typed programs and runs
each both with ordinary loops and with bounded scheduled unrolling. They cover:

- Finite cancellation, an independently calculated FMA product residual,
  smallest subnormal values, exact integer sums and signed zeros.
- Infinite or NaN inputs, `0 * infinity`, product overflow, high-prefix
  overflow, and opposite infinities. Finite-prefix overflow deliberately
  retains the serial high state's infinity even when the mathematical sum
  would later cancel to zero.
- Reading an intermediate compensated value, continuing an existing state,
  resetting through `Init`, replacing the base through `Accum.base`, seeded
  loops, empty seeded loops, and shuffle state preparation/combination.

Finite expectations use independent exact binary products and `math.fsum` for
the selected examples; expected nonfinite classes are explicit. Finite outputs
are compared by their FP32 bits, including zero sign. NaN and signed infinity
classes are checked separately. This is a focused semantic suite, not a full
floating-point proof by sampling.

The frozen baseline and candidate each passed **14 sm_89 assemblies and 14
local GPU numerical executions**, with no skipped methods. Reports
`guard-baseline-fixtures.json` and `guard-final-fixtures.json` record the
unchanged source hashes and exact fixture source. Local execution explicitly
adapts the target to the available sm_75 GPU; native scheduled asynchronous
copies are validated separately on the official runner.

## Performance interpretation

The [first paired runner study](../dot-guard-comparison.json) used six balanced
alternating rounds and twenty timed iterations per response. Each configuration
passed two normal seeds and one scaled-input seed before timing; every timed
response was checked against its independent output reference. All 72 responses
passed across the four workloads.

| Workload | Original guard, median us | Deferred guard, median us | Reduction |
|---|---:|---:|---:|
| m01 large matmul | 97.992 | 90.760 | 7.38% |
| m02 split-K matmul | 5.064 | 5.040 | 0.47% |
| a01 attention | 285.328 | 260.496 | 8.70% |
| a02 attention | 158.224 | 156.232 | 1.26% |

These are medians of the six response medians, measured on the official RTX
5090 runner with unlocked clocks. The split-K result is effectively unchanged
within its observed variation. They are indicative paired measurements for
these concrete sources and inputs, not guaranteed gains for every kernel or
hardware target. The
[additional-workload study](../dot-guard-additional.json) passed another 72
responses: RMSNorm 5.208 to 4.888 us (6.14% lower latency), L2 normalization
3.008 to 3.008 us, long-reduction matmul 19.104 to 19.096 us (effectively
unchanged), and irregular attention 53.920 to 51.856 us (3.83% lower latency).
Together these studies contain 144 passing responses. These are development
workloads, including a reused earlier corpus, not hidden evaluation data. The
candidate was selected for production validation; complete correctness gates
are recorded separately in the repository validation report.

`guard-static/summary.json` records both emitted PTX and CUDA 12.6 sm_89 machine
code. Static PTX may contain more finite tests because a consumer can reference
the same completed carrier several times. Their positions and dynamic loop
execution counts matter more than the static number. Likewise the assembler
may choose different unrolling after the dependency changes. Only the paired
correctness-qualified runner experiment can establish a runtime benefit.

The candidate uses no reciprocal approximation, precision downgrade, dot
striping, arithmetic reassociation, instruction backend, schedule change or
problem-name dispatch. The production strict policy remains byte-identical
where no compensated state is present.
