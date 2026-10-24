# Numerical policy and performance experiments

The compiler's default favors correctness coverage: exact/missing tolerances
retain ordinary arithmetic; nonzero tolerances enable compensated independent
FP32 dots. Bounded unrolling follows schedule hints. This policy addresses an
observed cancellation failure, but it has a substantial attention cost.

The continued **23 October 2026** audit selected backend SHA-256
`8afa55d5ac32da2eda6184058de9aa40ee8327366fc9fa6261b82fb0ed2ce002`.
It moves the finite guard for a compensated dot to state observation and retains
the same arithmetic. The earlier `0373bbc6...` release and historical experiments
remain recorded below. Current correctness gates are maintained separately in
[VALIDATION.md](VALIDATION.md); timing success does not replace them.

## Current paired experiment: defer the Dot2 finite guard

The original Dot2 loop discarded its correction whenever the high carrier became
nonfinite. The selected implementation stores the correction and, whenever the
carrier is read, returns `high + (isfinite(high) ? correction : +0)`. The rounded
product, FMA residual, full TwoSum, correction additions and iteration order are
unchanged. This removes a guard from each accumulation while preserving the
original materialized `high + 0` operation for nonfinite states.

This relies on a specific invariant: implicit additions cannot make a nonfinite
high finite again, and every explicit replacement either clears or recreates its
correction. The review covered `seed`, `Init`, `write_state`, `prepare_state`,
shuffle and tree paths. Seven independent typed fixtures, each with ordinary and
unrolled loops, checked cancellation, product residuals, subnormals, nonfinite
classes, overflow, observations, continuation, explicit base replacement,
resets, empty loops and shuffle combination. Both frozen implementations passed
14 assemblies and 14 local numerical executions. See the
[proof and focused reports](../artifacts/validation/phase3/performance/GUARD_DEFERRAL.md).

The [paired comparison](../artifacts/validation/phase3/dot-guard-comparison.json)
uses unchanged frozen compiler CLIs: baseline `6adcba46...` and candidate
`8afa55d5...`. `tools/compare_compilers.py` checks normal seeds 0 and 17 and scaled
seed 101, then performs six balanced alternating rounds with three warmups and
20 timed iterations per response. Every returned response is checked against
its independent reference; outputs are not checked after each repeated launch
inside a response. All **72 responses passed** for the four workloads below.

| Workload | Original guard, median us | Deferred guard, median us | Latency reduction |
|---|---:|---:|---:|
| m01, large matmul | 97.992 | 90.760 | 7.38% |
| m02, split-K matmul | 5.064 | 5.040 | 0.47% |
| a01, attention | 285.328 | 260.496 | 8.70% |
| a02, attention | 158.224 | 156.232 | 1.26% |

These are medians of the six response medians on the RTX 5090 runner with
unlocked clocks. Split-K was effectively unchanged within the observed
variation. These measurements establish a benefit for the recorded cases and
sources; they do not guarantee a gain for every workload or GPU. The additional
[normalization and generated-workload comparison](../artifacts/validation/phase3/dot-guard-additional.json)
uses the same protocol and frozen sources. All **72 additional responses passed**:

| Additional workload | Original guard, median us | Deferred guard, median us | Latency reduction |
|---|---:|---:|---:|
| f01, RMSNorm | 5.208 | 4.888 | 6.14% |
| f03, L2 normalization | 3.008 | 3.008 | 0.00% |
| Generated long-reduction matmul | 19.104 | 19.096 | 0.04% |
| Generated irregular attention | 53.920 | 51.856 | 3.83% |

The L2 and long-reduction results are effectively unchanged. These are
development performance/regression cases, including a reused earlier corpus;
they are not an untouched or organizer-hidden evaluation set. The two paired
studies together contain 144 passing responses across eight workloads.

The [static comparison](../artifacts/validation/phase3/performance/guard-static/summary.json)
shows why source size alone is insufficient. Large-matmul PTX decreased from
272,949 to 252,295 bytes and its finite tests from 256 to eight. However, the
first attention kernel's static sm_89 machine-code instruction count increased
from 1,392 to 1,544 as the assembler chose different expansion. Its measured
runtime still decreased. These static counts include padding/helper code and
are not dynamic instruction counts. In normalization, more static consumer
guards likewise replace guards inside repeated loops. Nine public programs
without compensated carriers emit identical PTX under this change.

The paired harness preserves submitted PTX bytes and hashes compiler sources,
inputs, references and harness files before and after execution. Its CLI
compilation times include Python startup/imports and exclude downstream
assembly. This differs from the older in-process search timings below.

## Rejected default: automatic scalar-load packing

An isolated generic candidate combined existing, aligned scalar global loads
into vectors without crossing writes or control flow, changing index
definitions, or moving the visibility of value bindings. It reduced public
pointwise sm_89 global-load instructions and register counts: p00/p01 had four
loads reduced to two and 16 registers reduced to 12; p03 had eight loads reduced
to two and 22 registers reduced to 20. Nineteen independent fixtures checked
alignment, tails, scope, dependency changes and conservative declines.

The [six-workload paired study](../artifacts/validation/phase3/load-packing-comparison.json)
passed all **108 responses**, but the median times showed no broad improvement:

| Workload | Baseline us | Packed loads us |
|---|---:|---:|
| p00 | 2.992 | 3.008 |
| p01 | 3.016 | 3.008 |
| p02 | 3.128 | 3.072 |
| p03 | 2.992 | 3.016 |
| Generated 262,144-element add | 3.040 | 3.056 |
| Generated 1,048,576-element ReLU/product | 4.800 | 4.832 |

The small p02 reduction did not justify adding this transformation to the
production compiler. The [candidate and proof](../artifacts/validation/phase3/performance/LOAD_PACKING.md)
are retained as a rejected experiment. An audit during this work found a
separate, real fixed-width divisibility bug in the baseline alignment analysis;
that correctness fix was retained and tested. The rejected packing snapshots
precede that fix and must not be copied into production.

The first packing report's PTX hash fields described saved files whose Windows
text write had duplicated carriage returns, rather than the posted payload.
The [payload audit](../artifacts/validation/phase3/load-packing-payload-audit.json)
retains the original report, reconstructs all 12 submitted payloads by
deterministic re-emission from unchanged snapshots, verifies the exact newline
relationship, and records separate payload hashes. It also identifies the
archived helper imported before an unrelated local-assembler diagnostic edit.
The guard experiments use the corrected byte-preserving harness.

## Historical configuration search

`tools/tune_lowering.py` compares explicit configurations using the same IR,
launch, references and inputs. Each candidate must assemble and pass seeds 0 and
17 with normal inputs plus seed 101 with inputs scaled by eight. Only candidates
that pass every check enter timing. Each timed round uses three warmups and 20
iterations; three rounds alternate candidate order. Reported values below are
medians of the runner's round medians, in microseconds.

The runner is an RTX 5090 with unlocked clocks. These are indicative development
measurements, not stable leaderboard scores or a claim about unseen shapes.
Experiments ran at different compiler revisions; each JSON records the backend
hash, emitted PTX hash, configuration, compilation time, code size, register
count and correctness. Do not treat cross-file differences as a precise paired
benchmark. Compilation time excludes Python startup/imports and assembler time.

| Workload | Strict loops | Compensated, bounded unroll | FP64, bounded unroll |
|---|---:|---:|---:|
| m01, large matmul | Rejected: scaled-input mismatch | 96.928 | 403.856 |
| m02, split-K matmul | 5.056 | 4.912 | 13.168 |
| a01, attention | 98.992 | 285.184 | 984.784 |
| a02, attention | 58.016 | 156.288 | 207.456 |

Sources: [strict/FMA study](../artifacts/validation/lowering-search.json),
[FP64 study](../artifacts/validation/accurate-dots-search.json),
[compensated study](../artifacts/validation/compensated-dots-search.json).
The strict large matmul measured roughly 100 us in the earlier, shorter public
check, but it failed the later stress gate. That historical measurement is not a
correctness-qualified timing comparison.

## Decisions

**Compensation instead of FP64 by default.** Dot2 addresses the observed
cancellation with FP32 arithmetic and FMA product residuals. It is much faster
than widening on these matmuls. Attention slows relative to strict FP32 by about
2.7-2.9x; that is an explicit cost, not a claimed optimization of every workload.
The mathematics and assumptions are in
[Ogita, Rump and Oishi](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).

**Retain bounded scheduled unrolling.** On compensated m01, the measured median
changed from 139.920 to 96.928 us. PTX grew from 15,677 to 272,949 bytes, and the
reported registers grew from 36 to 44. Compile time in those trials grew from
about 32 to 134 ms. The size budget is therefore important even when runtime
improves. The public attention loops were not expanded under this policy.

**Reject automatic expansion of unmarked dot loops.** A later experiment expanded
short independent dots even when the upstream hint was false. Attention a01
grew from about 15.7 KiB to 262.6 KiB of PTX without a useful runtime improvement;
a02 likewise showed no material gain. This experiment was removed from the
default. See [retained evidence](../artifacts/validation/bounded-dot-unroll.json).

**Reject fusion of adjacent independent dot loops.** The 23 October experiment
combined a01's four eligible same-bound dot loops while preserving each
carrier's iteration order. Both configurations passed the ordinary and scaled
input gate. Nevertheless, the paired median increased from **283.264 to
463.392 us**, despite the runner reporting **64 to 56 registers** and PTX
shrinking from 15,711 to 15,228 bytes. Register count alone did not predict
runtime. The option was removed from production; a02 and the tested matmuls
had unchanged emitted PTX, so incidental timing differences there establish no
benefit. See the [complete experiment](../artifacts/validation/phase2/loop-fusion.json)
and [rejected source archive](../artifacts/validation/phase2/rejected-loop-fusion.zip).

**Retain exact raster simplification.** If the second raster extent is one,
the mapping is the identity. When the first extent is an integer multiple of
the clamped group size, the group width is constant; otherwise the existing tail formula
is retained. Eight independent fixtures cover identity, full, clamped and tail
groups in both orientations. Six simplified fixtures lose a redundant
`min.s32`; the two tail cases retain byte-identical PTX. Public a02 PTX shrank
from 3,793 to 3,412 bytes in this isolated comparison. These are static and
correctness results, **not a measured runtime speedup**. The
[comparison record](../artifacts/validation/phase2/raster-simplification/ptx-comparison.json)
preserves both versions.

**Keep FMA contraction as a development alternative.** It slightly reduced the
measured attention times, but did not resolve the scaled m01 failure. It is not
the default and must never be introduced indiscriminately into exact pointwise
arithmetic or coupled recurrences.

No policy was trained. This is a deterministic grid and a documented engineering
decision. A cost model or contextual selector would need a larger structurally
diverse corpus, an untouched evaluation split, stable measurements and a gain
over simple rules at a matched evaluation budget.

## Earlier frozen generalization check

This section preserves the phase-2 evidence for backend `0373bbc6...`. The same
corpus is now reused for regression and selected performance experiments, so
the current iteration does not claim it as untouched evaluation. Current-source
reruns are recorded in [VALIDATION.md](VALIDATION.md).

The [18-workload protocol](HOLDOUT_2026_10_23.md) freezes new shapes and
compositions across the five categories before GPU evaluation. Workload source
definitions were visible during preparation; the candidate and policy were not
changed in response to evaluation outcomes. It is a
synthetic evaluation corpus, not the organizer's hidden tests. Two normal seeds
and one scaled seed give **54/54 passing cases for the production policy**;
strict arithmetic also passed **54/54 cases in the same frozen backend**. The
comparison is a numerical-policy ablation: the old historical source snapshot was unavailable
and is not reconstructed by disabling compensation.

The [candidate snapshot](../artifacts/validation/phase2/release-candidate/candidate-source.json)
records source and corpus hashes. Evaluation is a correctness gate with one
warmup and three iterations; incidental timings are not an interleaved
performance comparison. The [production-corpus report](../artifacts/validation/phase2/release-holdout-default/summary.json)
records all 54 passes; the complete phase-2 production gate was 157/157 cases.
The [strict-ablation report](../artifacts/validation/phase2/release-holdout-strict/summary.json)
therefore establishes no additional binary-correctness wins from compensation
on this corpus. Both policies passed its fixed tolerance checks; that does not
prove identical error or erase the retained earlier strict-matmul failure.
The ablation's incidental timings do not establish a policy speedup.
If a later change responds to these outcomes, the corpus becomes a
development set and cannot continue to support claims of untouched evaluation.

The same release includes correctness fixes for typed integer constants and
casts, tiny `sin`, stable `tanh`, signed zero and special values. Those changes
are covered by independent numerical-edge fixtures; no speed benefit is claimed
for them. See [SEMANTICS.md](SEMANTICS.md) for their domains and limits.

## Reproduce

Recheck the selected guard change using the frozen paired snapshots:

```console
python tools/compare_compilers.py m01_matmul_512 m02_matmul_rect a01_attention a02_attention_128 --baseline artifacts/validation/phase3/performance/guard-baseline/compile.py --candidate artifacts/validation/phase3/performance/guard-candidate/compile.py --rounds 6 --iters 20 --report artifacts/validation/guard-recheck.json
```

The older arithmetic-policy grid remains available for separate development
experiments:

```console
python tools/tune_lowering.py m01_matmul_512 m02_matmul_rect a01_attention a02_attention_128 --configurations strict_loops,strict_unroll,compensated_loops,compensated_unroll --rounds 3 --iters 20 --report artifacts/validation/new-lowering-comparison.json
```

`accurate_loops`, `accurate_unroll`, `fma_loops` and `fma_unroll` are also available.
All variants are implementations in this backend. The production CLI runs no
online autotuning, network calls or learned inference. Future experiments should
report worst regressions, failure counts, code/compile growth and uncertainty,
not only aggregate speedups among surviving candidates.

