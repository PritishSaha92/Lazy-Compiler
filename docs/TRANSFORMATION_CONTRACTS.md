# Conditions for lowering and optimization

The submitted compiler translates scheduled typed IR. The schedule, ABI and
cooperative participants are inputs, not tuning parameters. Optimization choices
must meet the conditions below before performance can influence their selection.
Correctness tests are evidence about cases; they do not prove universal legality.

| Choice | Preconditions and preserved behavior | Fallback and evidence |
|---|---|---|
| Static loop unrolling | Only a plain `Loop` with an upstream unroll hint; a static extent; per-loop structural work within the chosen budget and remaining whole-kernel budget. Emit iterations in their original order, retaining `seed`, nested initialization, branches, copy groups, waits and barriers. Fresh lexical scopes and labels preserve value availability. | Emit the original loop. Actions 0/256/1024 change the per-loop budget only; the whole-kernel cap remains 4096. The selector study checks every variant before timing. |
| Grouped memory instructions | Consecutive elements of the same buffer/type and a proven address alignment for the emitted width. Integer divisibility facts must remain valid after fixed-width wrap, division and casts. The complete access remains inside its original guard. | Scalar loads/stores in the same order and scope. Ragged-row tests exercise different source/destination pitches, grouped accesses and scalar cleanup. |
| Raster simplification | Algebraic identity or a proven fully populated/clamped group, using actual static dimensions and orientation. Physical block ownership and logical indices remain identical, including tails. | Original quotient/remainder mapping. Independent rectangular/raster fixture oracles check ownership. |
| Predicate promotion | Logical operators retain predicate results. C integer arithmetic on comparisons promotes operands to signed 32-bit values, before usual arithmetic conversions. | Diagnose an unsupported operator/type pair rather than invent a PTX spelling. Mixed-rank and predicate-expression fixtures distinguish the cases. |
| Narrow floating-point arithmetic | Expansion of FP16/BF16 operands to FP32 is exact. A declared narrow `Assign` result uses the existing FP32 operand-conversion/computation policy and narrows at the statement boundary; wider operands can round during conversion. Each narrow `BinaryExpr` separately rounds its result. Do not move rounding past another narrow expression node. This is an explicit backend numerical contract, not a correctly-rounded guarantee for every transcendental or possible double-rounding boundary. | Reject unsupported types or target-dependent conversions. The capability matrix, expression-form fixtures and conversion tests record admitted paths and limitations. |
| Compensated independent dots | Nonzero metadata tolerance, and the existing independence analysis excludes observed carriers, rescaled bases and coupled state. Preserve the compensated algorithm's update sequence and carrier lifetime. | Ordinary arithmetic for exact/missing tolerances or ineligible carriers. This changes source rounding and has empirical tolerance evidence, not an all-input error proof. |
| Deferred Dot2 finite guard | Finite update sequence is unchanged. A nonfinite high cannot become finite by additive updates; replacing a carrier resets its correction. At consumption select zero correction when high is nonfinite, and retain `high + 0`. | Materialize the required value; never propagate a stale correction across assignment or a new lexical carrier. Existing recurrence/nonfinite fixtures and retained paired timings cover this change. |

For any future rewrite, record: typed node form; operands and lexical scope;
integer width/range and overflow assumptions; guard and value dominance;
numerical order and rounding; memory effects; synchronization participants;
ordinary fallback; an independent counterexample; source-matched evaluation.
Changing a barrier, copy wait, atomic reset, launch geometry, or precision is a
semantic change requiring its own justification, not another harmless search arm.

The development analysis command is:

```console
python tools/analyze_ir.py problems/m01_matmul_512/k_matmul_2967be.ir.json --output build/analysis.json
```

Use the actual IR filename in the problem directory if it differs. The sidecar
reports structural counts, declared/aligned shared allocation and memory access
sites. It does not prove bounds, arbitrary alignment or coalescing. Optional
`--emit --assemble` adds own-emitter size/latency and downstream `ptxas` resource
data without executing a GPU. Static access-site bytes are not measured DRAM
traffic; virtual register declarations are not physical register pressure.
Neither the analyzer nor the selector is imported by production compilation.
