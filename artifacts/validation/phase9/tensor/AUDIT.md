# Bounded tensor fragment extension

The previous release already lowers m16n8k16 FP16/BF16 MMA with FP32 accumulators,
global/shared operand loads, guarded stores and two-C-fragment-to-A repacking.
Two missing pinned-IR operations prevented the existing fragments from expressing
row normalization without storing and reloading intermediate matrices:
`FragmentApply` and `FragmentRowReduce`.

The source of semantics is the installed Emmy 0.3.11
`emmy/compiler/ir/kernel/ir.py`, specifically `M16N8` (around line935),
`FragmentApply` (993-1171), and `FragmentRowReduce` (1175-1226). The public
`vguerra__emmy-hackathon-2026/COMPILER.md` snapshot lists the same operation family,
which confirms its relevance as a scope comparison. Its implementation was not
executed or copied. This backend is an independent translation of the pinned IR.

`FragmentApply` now accepts all five distributions: fragment elements, row scalar
pairs, named/literal uniform scalars, coordinate predicates and global memory
templates. It supports fresh outputs, in-place updates and existing dynamic family
indices. Unary post operations use the scalar backend. The `where` form emits its
operand loads inside the selected branch: a coordinate predicate can safely mask
an otherwise out-of-range global address. Missing origins fail explicitly.

The admitted layout is exactly M16N8, with four FP32 elements per lane. Row folds
visit fragments in order and elements in their defined order, then shuffle with
XOR masks2 and1. Add, multiply, minimum and maximum are admitted. Collectives
require complete warps, unconditional ancestry and static loop controls, with no
leaf rebinding of an enclosing induction variable. Conditional/dynamic control is
rejected rather than using an unsafe full membership mask. Other layouts and
FP16 accumulation remain outside this extension.

An explicit mixed-precision boundary prevents early rounding: pinned Emmy narrows
named UNIFORM and GMEM values to FP32, but lets ROW and expression operands retain
their expression type. Consequently f64 ROW, numeric COORD and UNIFORM Expr
results are rejected. A predicate computed from double inputs remains supported.

Eight original runner packages live in `examples/phase9/fragments`:

- A masked, offset global gather with an in-place uniform scale and unary post.
- Two side-by-side fragments normalized through live family indices.
- FP16 MMA followed by row maximum, exponential, sum and division.
- A single 16x16 attention tile: two QK atoms, combined normalization, FP16 weight
  repack, and PV. Inputs are bounded dyadic values; the host reference explicitly
  rounds weights to FP16. This is not a claim of a complete attention algorithm.
- Exact ordered addition, multiplication, maximum and minimum row reductions.

The six focused unittest methods passed, including actual CLI compilation and
ptxas sm_89 assembly for all eight cases. The tests also cover malformed operands,
missing coordinate origins, partial warps, conditional/dynamic collective
rejection, unsupported layouts and f64 expression rejection. The addition oracle
uses cancellation-sensitive data and an independent lane ownership simulation;
the normalization references use independent matrix formulas. No local GPU was
used. Organizer-runner results are separate evidence owned by the parent task.

The first runner attempt rejected the attention and ordered-add packages before
device execution because their explicit JSON data arrays were nested. The
corrected metadata flattens those lists without changing values, order, shapes,
IR or expected output bytes. Preserved originals remain under
`artifacts/validation/phase9/initial-fixtures`. `transport-fix.json` proves that
flattening is the only metadata change and records the unchanged IR/NPZ hashes.
The final six-method CPU/CLI/assembly suite passed again after this correction.

`check_compatibility.py` compares the current compiler output against the phase8
recorded hashes for all15 public modules and41 existing feature modules. Exact
bytes are compared, including launch bounds. The report records source hashes and
whether sources stayed unchanged during compilation.
