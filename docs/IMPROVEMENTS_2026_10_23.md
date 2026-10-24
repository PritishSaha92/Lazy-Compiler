# Continued compiler review and optimization

This round began by inspecting the user's `out.ptx`, then reviewing required IR
semantics independently of the public cases. That file is the complete scalar-add
kernel for one input IR, not the whole compiler. Its 64 valid work cells each add
two elements; the bounds guard excludes the other threads in the prescribed
256-thread launch. Repeated address arithmetic in the PTX mostly disappears in
the downstream assembler. Machine code and numerical execution, rather than PTX
length, determine whether a change helps.

The previous backend (`0373bbc64340...`) passed another **60 official-runner
cases**: all fifteen open workloads at seed offsets 233 and 419, with normal and
scaled inputs. Nevertheless, independent minimal programs exposed four defects.
This illustrates why repeated public runs alone cannot establish generality.

## Correctness fixes

| Defect | Required behavior and fix | Evidence |
|---|---|---|
| Mutable strided-loop bound | `StridedLoop.end` is captured once, including its `<axis>_end` binding. A reference to a carrier previously remained live and changed the loop termination condition. Snapshot it before entering the loop; the step expression remains live. | Three programs cover changing bounds, nested recapture, captured starts and changing steps. The old implementation failed 36 of 51 elements in the first fixture. |
| Legacy partial-coordinate indexing | Emmy's installed indexing helper sums coordinates when a tuple has more than one coordinate but its rank differs from the buffer's. The backend rejected this form. Implement the same sum, using widened arithmetic for large buffers. | Three numerical programs cover mismatched loads and stores; host checks cover offsets beyond signed 32-bit range. |
| Arithmetic on predicates | Two comparison results used by arithmetic or bitwise `BinaryExpr` must promote to integer, as in the installed C-style expression semantics. The backend could emit instructions such as `add.rn.pred`, which do not assemble. | One fixture exercises twelve operators. The original assembler errors are retained. Logical `&&` and `||` retain predicate lowering. |
| Alignment facts after integer overflow | A divisibility fact must survive fixed-width wraparound before it can justify a vector memory operation. Intersect runtime add/subtract/multiply facts with the integer width's modulus. | Four programs cover scalar and grouped loads with 32- and 64-bit indices. Unsafe old PTX was inspected but deliberately not executed. Corrected programs assemble and return exact references. |

For the alignment counterexample, let unsigned 32-bit `x = 1431655766`.
Fixed-width `x*6` wraps to four, so `(x*6)/3` is one. The old analysis retained a
factor of six and then inferred a factor of two after division. That incorrectly
authorized an eight-byte vector load at an odd FP32 element offset. The corrected
analysis keeps only facts true after wrapping; no extra memory accesses or
runtime checks are introduced.

Detailed contracts are in [SEMANTICS.md](SEMANTICS.md). Retained failures and
fixed checks are under [phase3](../artifacts/validation/phase3/), especially
`loop-capture-baseline.log`, `index-layout-baseline.log`,
`predicate-promotion-baseline-assembly.json`, and
[`alignment-overflow/baseline-proof.json`](../artifacts/validation/phase3/alignment-overflow/baseline-proof.json).
The final release gate, rather than these intermediate checks, determines the
current source's complete coverage.

## Rejected default: combining scalar global loads

An isolated candidate combined unconditional consecutive scalar loads into
aligned two- or four-element loads. It crossed only pure scalar bindings,
preserved index definitions, and delayed each loaded name's binding until its
original program position. Stores, barriers, accumulator updates and control
flow ended the search. Nineteen independent fixtures exercised alignment,
tail widths, index/name rebindings, memory updates and conservative declines.

The candidate reduced machine-level global-load instructions and registers on
the four public pointwise kernels under local `sm_89` assembly. However, six
alternating rounds of twenty timed iterations on the official GPU did not
establish a reliable overall runtime benefit:

| Workload | Baseline median, microseconds | Candidate median, microseconds |
|---|---:|---:|
| Public scalar add | 2.992 | 3.008 |
| Public vector add | 3.016 | 3.008 |
| Public scale/shift | 3.128 | 3.072 |
| Public ReLU/product | 2.992 | 3.016 |
| Generated 262,144-element add | 3.040 | 3.056 |
| Generated 1,048,576-element ReLU/product | 4.800 | 4.832 |

These are medians of per-response medians, with unlocked clocks. There were
36 correctness responses and 72 timed responses; all 108 returned outputs passed
their reference checks. The API returns outputs once per response, so this does
not mean every internal timed repetition was separately checked. Paired-round
ratios also showed no convincing overall gain. The compiler therefore does not
enable this additional transformation. Its source, proof, tests and measurements
remain available for future architectures or workloads.

See the [comparison](../artifacts/validation/phase3/load-packing-comparison.json),
[candidate proof](../artifacts/validation/phase3/performance/LOAD_PACKING.md), and
[payload audit](../artifacts/validation/phase3/load-packing-payload-audit.json).
The original experiment's Windows text writer duplicated carriage returns in
saved PTX artifacts; the posted payload was unaffected. The audit reconstructs
and hashes all twelve submitted payloads from the unchanged frozen compilers,
while retaining the original artifacts and report. The comparison harness now
writes exact bytes, records paired ratios, rejects invalid timings/truncated
outputs, and checks source/reference/input stability.

## Selected improvement: compensated-dot guard deferral

The selected improvement moves the correction's finite-value guard from
every Dot2 update to materialization of the result. The rounded product, FMA
product residual, full TwoSum, and low-state addition remain in their original
order. An independent dot's high state changes only through rounded addition;
after becoming infinite or NaN, it cannot become finite without an explicit
replacement. Seeds, initialization, ordinary updates and cooperative state
replacement reset or recreate the correction.

Consequently a finite high state observes exactly the original low state. For a
nonfinite high state, lookup selects a zero correction and then performs the
original `high + 0` operation. Internal invalid correction values are not exposed.
This is an execution-cost experiment, not a weaker numerical policy or an
approximate replacement for compensation. Seven new fixtures test exact finite
bits, nonfinite classes, snapshots, continuation, resets, zero-trip loops and
cooperative reduction with retained and unrolled loops. Both frozen variants
passed fourteen assembly and fourteen local numerical checks before timing.

Across eight workloads, all 144 paired-runner responses passed their output
checks. Six alternating rounds of twenty timed iterations measured median
latency reductions of 7.38% for the large matmul, 8.70% for the larger attention
case, 6.14% for RMSNorm, and 3.83% for irregular attention. The other measured
workloads were close to unchanged. Clocks were unlocked, so these are measured
improvements for this comparison, not guarantees across runs or hardware.

The selected backend is `8afa55d5ac32da2eda6184058de9aa40ee8327366fc9fa6261b82fb0ed2ce002`.
Promotion and source-matched release results are recorded in
[VALIDATION.md](VALIDATION.md) and [PERFORMANCE.md](PERFORMANCE.md).

## Additional coverage

[PHASE3_CASES.md](PHASE3_CASES.md) records nine transport fixtures, two exact
product reductions, five new schedules and independent CPU checks. The new
schedules include non-innermost softmax, centered column normalization, batched
ragged matmul, irregular attention and three-dimensional broadcasting. Emmy's
failure to trace a sixth planned product expression is retained separately;
it is neither a passing backend case nor a backend failure. These are our
development cases, not the organizer's undisclosed tests.
