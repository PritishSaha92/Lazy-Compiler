# Scalar-load packing experiment

**Disposition: rejected as a production default.** Six balanced runner rounds
of twenty iterations each found no broad runtime benefit across the four open
pointwise workloads and two larger generated workloads. p02 improved slightly,
while the other medians were effectively flat or slightly worse. The static
instruction reduction alone does not justify adding this optimization to the
production compiler. The isolated candidate and independent tests remain as
development evidence. A subsequent audit found and fixed an existing integer
divisibility fact overflow issue in production; the rejected snapshots precede
that fix and must not be promoted as-is.

The [complete paired study](../load-packing-comparison.json) retains all 108
correctness-qualified responses. Median microseconds changed as follows:
p00 2.992 to 3.008; p01 3.016 to 3.008; p02 3.128 to 3.072; p03 2.992 to 3.016;
262,144-element add 3.040 to 3.056; 1,048,576-element ReLU/product 4.800 to 4.832.
These unlocked-clock results do not establish a broad speed improvement.

This candidate combines existing scalar global loads into aligned PTX `v2` or
`v4` loads. It preserves the scheduled thread ownership and arithmetic order.
The production compiler is unchanged by the experiment builder. Frozen compiler
CLIs live in `baseline/` and `candidate/`; the paired runner experiment owns the
timing decision. A later rebased candidate must preserve the six benchmark PTX
hashes before using those measurements as evidence.

## Scope and legality

The first implementation deliberately accepts only 32-bit source elements,
one flattened index, and ordinary global loads. Source annotations must be
absent or equal to the buffer element type. It matches structurally equal
expression trees with explicit consecutive integer offsets; it does not infer
equivalence from printed expressions or reassociate index arithmetic.

For an earliest scalar load, the scanner considers at most the next 31
statements. It may cross only `Load`, `Assign`, `Let`, and `Select` bindings.
Stores, accumulator updates, synchronization, asynchronous operations and all
nested control-flow constructs end the search. Therefore the move cannot cross
a write, change the participating threads, or speculate an access from a branch
that might not execute. Every vector component already has an unconditional
scalar load in the same straight-line region; no tail elements are added.

Every variable used by the original index must keep its definition throughout
the region. A definition by any intermediate binding, including a load, ends
the search. The first load is declined if it changes its own index dependency.
This handles mutable names as well as ordinary SSA input. Loaded values are
fetched together but assigned to the IR environment only when their original
load statement is reached. Consequently a repeated result name or a read of its
older binding sees the original value at the original program position.

The emitted index must have an integer type. Floating-point indices are
declined because separately converting `j` and `j+1` does not necessarily
produce consecutive integer addresses. Existing alignment facts must prove
8-byte alignment for two elements or 16-byte alignment for four. A four-value
group may reduce to two when only 8-byte alignment is known. If alignment is
unproved, scalar lowering remains in use. For valid integer addressing, an
aligned 2- or 4-element group cannot cross an integer wrap boundary within the
group, because those boundaries are multiples of the group size. As with the
existing emitter, this reasoning assumes valid input memory accesses.

Loads have no IR volatility/acquire modifiers. No arithmetic result is
reassociated or contracted. Compensation, reduction state, barriers, launch
geometry and ABI are unaffected. The compiler does not dispatch on problem
names, read reference outputs, access the runner or invoke another backend.

## Independent checks

`tests/test_load_packing.py` covers 19 typed cases:

- Widths two, three and four with aligned and shifted starts. The three-value
  case checks that packing two does not introduce a fourth access.
- A future load that shadows a previously read name; four packed loads reusing
  one result name; index dependencies reassigned by `Let`, `Assign`, or `Load`.
- A global store between loads, including a store in a conditional body.
- Fractional floating-point indices whose separate truncations alias.
- Missing, reversed and duplicated offsets.
- Inconsistent source type annotations, retained as conservative decline tests
  rather than interpreted as a new casting feature.

Each numerical result is checked exactly against independently constructed
NumPy arrays. `run_fixtures.py` imports an isolated frozen backend and records
its source hash before and after assembly and local GPU execution. Local
execution adapts the PTX target explicitly to the available `sm_75` device.

## Static observations

For the initial public-case comparison, CUDA 12.6 `ptxas -arch=sm_89` emitted:

| Public workload | Scalar/global load instructions before -> after | Registers before -> after |
|---|---:|---:|
| p00 and p01 | 4 -> 2 | 16 -> 12 |
| p02 | 6 -> 3 | 24 -> 19 |
| p03 | 8 -> 2 | 22 -> 20 |

The other eleven public PTX outputs were unchanged. These are static target
observations, not measured speedups. All four input values of p03 become two
128-bit loads. The original `out.ptx` has no spills, and the assembler already
removes its redundant integer calculations; changing PTX text alone would not
establish better machine code.

Two additional workloads in `bandwidth/` were generated by the pinned Emmy
0.3.11 scheduler: a 262,144-element add and a 1,048,576-element ReLU/product.
Their mathematical definitions and independent NumPy references are retained
in `expressions_bandwidth.py`. They are development performance cases, not an
untouched or organizer-hidden evaluation set. The larger arrays help reveal
memory-instruction effects that may be hidden by launch overhead in p00.
