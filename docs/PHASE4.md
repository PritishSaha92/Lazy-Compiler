# Phase 4: typed lowering and verified release

The frozen compiler passes **236 default and 6 explicitly configured variant
numerical checks across 120 distinct development workload packages**. Its
three-file submission archive has been built, inspected and exercised through
the real extracted command-line entry point. The
[aggregate release summary](../artifacts/validation/phase4/release-summary.json)
records the source, report and artifact hashes. These are development results;
the organizer's hidden test set remains unavailable.

## Compiler changes

The research review and source audit identified instruction admission and
interactions between types, scopes, tails and cooperative memory as useful next
work. Phase 4 makes the accepted scalar lowering paths explicit and adds
independent numerical fixtures. It retains the prescribed launch metadata,
scheduled work ownership and existing FP32 arithmetic policy.

| Change | Result and correctness boundary |
|---|---|
| Operation admission | `_ELEMENTWISE_CAPABILITIES` records supported operation/result-type pairs and operand counts before an instruction is formatted. Conversion, comparison, bitcast-width and target checks remain separate. Unsupported combinations raise contextual `CompileError`; failed CLI compilation emits no partial PTX. |
| Narrow binary expressions | FP16/BF16 operands expand exactly to FP32, the binary operation executes, and its result rounds back at each narrow expression node. A surrounding FP32 `Let` cannot erase an intermediate narrow rounding boundary. This fixes previously invalid instruction spellings such as `div.rn.f16`. |
| Declared narrow statements | `Assign` and the optional `FuncCallExpr` extension use the existing FP32 computation policy and explicitly narrow the result. Conversion of wider operands to FP32 can itself round; this is not a claim that every composite is a correctly rounded native half operation. |
| BF16 comparisons and conversions | Comparisons expand to FP32 instead of requiring native `sm_90` BF16 comparison instructions. FP32/BF16 numeric conversion requires `sm_80+`; exact intermediates are used where justified. FP64 and 32/64-bit integer conversion to BF16 requires native `sm_90+` support, avoiding an unproved FP32 double-rounding fallback on `sm_89`. |
| Narrow selection and bitcasts | Small integer selections use legal 32-bit register operations and explicit narrowing. Narrow payload bitcasts pack or unpack register lanes explicitly, preserving same-width payload bits and signed zero. |
| Unsigned unary operations | Unsigned negation uses subtraction from zero modulo the declared width; unsigned absolute value is identity. These replace invalid unsigned `neg`/`abs` spellings. |
| Predicate behavior | Narrow floating comparisons and truth tests preserve unordered inequality and nonzero NaN behavior. Dedicated fixtures construct nonfinite values inside the kernel and return finite zero/one outputs. |
| Raw bit types and invalid combinations | Raw `b32`/`b64` values remain bit transport types. Numeric arithmetic requires an explicit same-width bitcast to a numeric type; unsupported floating remainder/bitwise expression forms fail with diagnostics instead of invalid PTX. |

Signed `bitwise_count` includes a deliberate semantic correction: it counts the
absolute value, so `i32(-3)` produces 2. Pinned Emmy's CUDA and loop render targets
instead map directly to raw population count, which would produce 31. The
chosen behavior follows the named NumPy operation. Minimum signed values retain
their single high bit through the magnitude operation and produce 1. Independent
I16, I32 and I64 cases cover this distinction. The installed NumPy predates that
named function, so the fixtures use `FuncCallExpr` and a Python
`abs(int(value)).bit_count()` oracle; they do not claim execution by the local
Emmy interpreter.

The complete policy is in [TYPE_SUPPORT.md](TYPE_SUPPORT.md), with transformation
preconditions in [TRANSFORMATION_CONTRACTS.md](TRANSFORMATION_CONTRACTS.md).
Before-change source, seven assembler failures and public PTX hashes are retained
under `build/type-capabilities/`. All fifteen public default PTX outputs remain
byte-identical across this change. This preserves their existing generated code;
Phase 4 does not claim a speedup from the type-admission work.

## Added validation coverage

The capability suite checks 55 operation names and **488 admitted
operation/result-type pairs** using downstream `ptxas`. It also checks conversion
and bitcast matrices, target-dependent rejection, wrong arity, and diagnostic-only
CLI failure. Assembly matrix kernels are never launched. Successful assembly
establishes instruction legality for those generated modules, not numerical
correctness for every admitted combination.

Nineteen independently checkable numerical packages cover FP16/BF16 binary,
statement and function forms, nested rounding, comparison/casting, nonfinite
predicates, bitcast round trips, unsigned unary operations, small integer
selection and signed population counts. Their API buffers use explicit FP32
data while typed intermediates exercise the narrow lowering paths. This avoids
relying on BF16 transport support in the runner API; it is not a substitute for
testing BF16 storage transport.

Four interaction packages add:

- A guarded load whose fallback must preserve previously live state.
- Nested row/tail guards with state carried at the correct scope.
- Ragged inner dimensions with row-dependent alignment and scalar cleanup.
- Seven asynchronous stages reusing two shared-memory slots with cross-warp
  consumers.

These 23 packages pass with the default compiler. Six additional checks cover
all four interaction packages with rolled loops, plus the asynchronous case
with synchronous-copy diagnostic lowering in both rolled and unrolled forms.
Those wrappers preserve launch metadata and arithmetic order and are explicitly
recorded as variants; they are not shipped in the submission ZIP.

## Final evidence

| Gate | Verified result | Retained report |
|---|---|---|
| Broad organizer execution | 213/213 default numerical checks; 97 workload packages; source and tracked inputs/helpers unchanged | [release-remote summary](../artifacts/validation/phase4/release-remote/summary.json) |
| Added semantic execution | 23/23 default plus 6/6 variant checks; 23 additional packages; tracked sources unchanged | [semantic-final summary](../artifacts/validation/phase4/semantic-final/summary.json) |
| CPU and assembly regressions | 73 test methods: 55 passed, 18 explicit GPU skips, no failures/errors; no GPU execution requested | [CPU report](../artifacts/validation/phase4/cpu-regressions.json) |
| Clean installation and CLI | Seven packages, each compiled twice normally and once with audit guards: 21 successful invocations; three expected diagnostics; source unchanged | [clean-install summary](../artifacts/validation/phase4/clean-install/summary.json) |
| Public emission identity | 15 actual CLI PTX payload hashes agree with exported identities and organizer execution payload identities | [public identity report](../artifacts/validation/phase4/public-ptx-identity.json) |

The total is **242 numerical checks = 213 + 23 + 6**, of which **236 use the
default compiler**. The 120 distinct packages count workloads once; seeds,
input distributions and compiler variants do not add packages. Older candidate
or failed development reports remain historical evidence and are not included
in this total. Additional selector experiments are separate from these frozen
release gates.

## Frozen sources and release artifact

| File | SHA-256 |
|---|---|
| `compile.py` | `4ed3fb67833b7de20693c7cb3f163a4646cac8d3a71e63649c1af1bccd37ce56` |
| `ptx_backend.py` | `5cb3a351840ca03055b9725471fd3b44afee7a1d8e2832c49d439018da5ba9fc` |
| `requirements.txt` | `5d054f8757b1212865653105a4e7c634e2952007da020d2f67ad456eb6bccc84` |

The submission ZIP (`submission-bcbc4a8381497f1d34be.zip`, separate historical release artifact)
contains exactly these three files, with SHA-256
`b9af6a761233e39b8faa330e1fd01c49e8a457498cb5d3b7e2465ef5403251e8`.
The [manifest](../artifacts/release/phase4/submission-bcbc4a8381497f1d34be-b88459958203b4fdede2.manifest.json)
has SHA-256
`b88459958203b4fdede27e69a685283b4b67bfe26c20f9e7432c40a0479f486a`.
It retains exact snapshots of all five gate reports and their production-source
identities. [PACKAGING.md](PACKAGING.md) documents the actual release command.

An independent [post-package check](../artifacts/release/phase4/post-package-verification-b9af6a761233e39b8faa.json)
reverified the ZIP inventory, extracted source bytes and retained evidence,
then reran the extracted CLI. The p00 output was byte-identical to the built
artifact: 1,524 bytes, SHA-256
`c0d81f88369a245e4144ff5e364ed1139349e24a40f8b3ca26f6257b41729a41`.
The final organizer report records the same submitted PTX hash and size. No GPU
or network access was used by packaging or this post-package CLI check.

## Limits and pending work

The results support the implemented paths on the exercised workloads. They do
not establish correctness for every unseen graph, numerical input or target.
The narrow arithmetic policy retains existing FP32 composite approximations;
it is not a fully correctly rounded mathematics library. Target-dependent
conversion rejection is intentional. The type capability matrix has broader
assembly coverage than numerical execution coverage.

Organizer timings in these correctness reports are indicative, serialized
measurements, not paired performance evidence. The clean-install check uses a
fresh environment on the same Windows host and Python version; dependencies
other than Emmy are not hash-locked. Python audit hooks are regression checks,
not an operating-system sandbox against arbitrary native code.

## Separate selector feasibility result

The selector's [CPU preparation](../artifacts/validation/phase4/selector/preparation.json)
covered 30 workload packages and 90 action emissions, with 32 distinct PTX modules
passing assembly after deduplication. The
[feasibility report](../artifacts/validation/phase4/selector/feasibility.json)
compares all 90 configurations against the current production source and records
matching PTX identities. Its compiler hashes match the frozen release above;
the report SHA-256 is
`7d40a1cb8c2ba9c0b134fa953ce7003473635e82bd6bab3906a9e280c91dc4d4`.

| Split | Workloads | Workloads with different PTX across actions |
|---|---|---|
| Training | 10 | 1 |
| Validation | 10 | 0 |
| Final evaluation | 10 | 1 |

Both held-out splits fail the protocol's minimum of six workloads with distinct
action outputs. The recorded decision is therefore
`static_rejection_insufficient_nonidentical_actions`: retain the fixed 1024
unroll budget. No selector fitting or comparative timing was performed, and no
learned policy or RL improvement is claimed. Comparing actions that produce the
same PTX would measure timing variation rather than an optimization decision.

Correctness-only organizer runs for these selector packages are tracked
separately; assembly and PTX identity do not establish their numerical result.
They are excluded from the immutable five-report release manifest and the
242-check aggregate above. The aggregate retains its original pending selector
snapshot; this separate section records the later static decision without
rewriting that evidence artifact.

## Additional sealed-corpus correctness

After static rejection of the selector opportunity, the fixed production policy
passed **90/90 organizer-GPU checks across 30 additional packages** (normal seeds
0/17 and scaled seed 101). The [closed summary](../artifacts/validation/phase4/selector-correctness-retry01/summary.json)
records 30 assemblies, 90 returned-output checks, 120 HTTP attempts, no summary-write
retries, and unchanged compiler/tracked sources. The 90 action-identity comparisons
in the feasibility report link the frozen study payloads to this production source.

The original 242-check archive gates remain immutable. Combining those gates with
this separate correctness record gives **332 checks across 150 logical packages**:
326 default-policy checks plus six explicitly configured variants. The first
attempt stopped on a Windows summary-file replacement error after 22 recorded
passes; one further successful response survived in raw artifacts. That incomplete
attempt is retained and excluded from the total. The error's external cause is
not established, and it was not a reported kernel mismatch.

No policy was fitted and no comparative timing was performed. These new cases
strengthen development correctness coverage; they do not establish hidden-set
coverage or learned performance improvement. The
[final evidence index](../artifacts/validation/phase4/final-summary.json) binds the
immutable release and additional audit records without changing the ZIP.
