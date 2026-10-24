# Independent selector protocol review

Reviewed on 23 October 2026. This review covers the development-only unroll
selector in [selector_study.py](../tools/selector_study.py), the authored
[expression corpus](../tests/expressions_selector.py), and its
[focused tests](../tests/test_selector_study.py). The reviewed protocol is suitable
for a small, controlled pilot after the ordinary compiler correctness gate.
No selector performance improvement or production promotion is established by
this review. The later provenance and feasibility addendum below records the
completed preparation and static rejection before learning or comparative timing.

The reviewer read the implementation, checks and evidence; the implementation
owner executed the CPU tests. The final review did not run tests, notebook code,
or local or remote GPU work. Earlier review supplied the counterexamples below.

## Findings and repairs

| Finding | Why it mattered | Reviewed repair and evidence |
|---|---|---|
| Product oracle accepted an entirely zero result | The original `sigmoid(x).prod(-1)` used an absolute tolerance large enough to accept all-zero output on every canonical input for the larger shape. A passing check could therefore conceal a broken reduction. | The expression and independent NumPy reference now use `sigmoid(x * 0.125).prod(-1)`, with `atol=1e-12`, `rtol=1e-3`. `check_reference` explicitly raises if an all-zero implementation would pass. Both shapes, `(23, 11)` and `(65, 23)`, are checked for seed 0 normal, seed 17 normal, and seed 101 wide inputs. The focused test also verifies that deliberately weakening the tolerance triggers rejection. |
| Aggregate results were trusted without reconstructing transport evidence | Editing aliases, aggregate timings or event-to-round assignments could produce a plausible report disconnected from measured PTX or returned outputs. | `verify_measurement_evidence` rehashes every request, response and HTTP-attempt artifact, checks the final successful HTTP body against the response, reconstructs sealed PTX aliases and exact compile/run payloads, and recomputes every output check and timing summary. Each event must be used once. Compile events, input-major correctness checks, and balanced timing rounds must occur in increasing event-number order. Tests mutate each artifact type, aliases, a timing value and timing-event assignments. A compile response with the wrong entry point cannot proceed to execution. |
| Existing generated packages could survive a specification change | Reuse based on expression text alone could retain IR generated with old tolerances, scheduling knobs or source code. | `verify_package` compares expression and tolerances, a signature containing generation knobs, source/generator/worker hashes, and generated artifact hashes. `prepare` regenerates missing, stale or unproven packages before sealing. Tests change the tolerance, knobs and source independently. Generation requires `emmy-ml==0.3.11`. |

All three findings are closed in the source revision recorded below. The
chronological event check closes a follow-up issue found during the second
review: correct labels alone did not prove that the referenced measurements
actually formed the declared pairs.

The [focused test log](../artifacts/validation/phase4/selector-preparation/focused-tests.txt)
records **22 passing tests**. The
[repaired product evidence](../artifacts/validation/phase4/selector-preparation/product-oracle-repair.json)
records all six shape/input combinations passing the CPU PyTorch-versus-NumPy
check with zero-rejection enforced, and identifies the current expression-source
hash. This means an **entirely zero output is rejected for each case**; it does
not claim that every individual small output differs from zero by more than the
tolerance. The earlier 90-case CPU oracle audit predates this product repair and
is not evidence for the repaired expression.

## Protocol assessment

- **Separate model selection and final evaluation.** The corpus has 30 workloads,
  15 expression families and two shapes per family. Each split has ten workloads
  and five families, one per problem category. Expression and declared schedule
  families cannot cross splits; changing an input seed does not create a new
  held-out family. Feature normalization uses training data only. `fit` reads
  training and validation results, refuses existing final results or an existing
  selection, and freezes the models and selected policy before final collection.
  Final evaluation verifies the frozen hashes and cannot promote a different
  policy simply because it wins on final measurements.
- **Equal learning feedback, with explicit replay limits.** The tree and bandit
  each receive one baseline and one alternative aggregate observation per
  training workload: 20 logical observations each. A ledger reports physical PTX
  aliases separately. All configurations are measured once for the experiment;
  exposing only the recorded observations to each learner is offline replay,
  not evidence of on-policy RL or equal total collection cost to a deployable
  adaptive system. The fixed policy and predeclared heuristic are measured on
  the same workload records. The final oracle ceiling is explicitly
  nondeployable.
- **Deduplicate before measuring.** The actions change only the existing unroll
  budget to 0, 256 or 1024. Exact PTX-byte hashes identify aliases, favoring the
  baseline as canonical when present. Aliases share the same measurements; an
  unchanged PTX module cannot acquire an artificial gain from a separate noisy
  run. Launch metadata, numerical defaults and synchronization remain fixed.
- **Check outputs before and during timing.** Every unique PTX candidate must
  assemble with the expected entry point and pass all three explicit input cases
  before timing that workload. Returned outputs must be complete, finite, have
  the required names/shapes and satisfy the authored tolerances. Six balanced
  timing rounds use three warmups and twenty timed iterations per request. The
  checks cover every returned output, not every internal launch within a runner
  request.
- **Use paired uncertainty and a conservative gate.** The summary resamples
  expression families, then paired timing rounds within each workload. Promotion
  requires at least six workloads with nonidentical available PTX choices across
  two families, geometric-mean speedup of at least 1.03, lower 95% bootstrap bound
  strictly above 1, and no workload slowdown exceeding 5%. The nonidentical-choice
  count concerns available configurations, not a claim that the selected policy
  changes six workloads. Failed correctness/evidence checks block selection.
  A negative, noisy or insufficiently varied result keeps fixed budget 1024.
  A passing study only creates eligibility for separate review; it does not
  modify production compilation.

## Limits on conclusions

Five final families and six timing rounds are a small pilot. The family bootstrap
acknowledges related shapes but does not establish broad generalization, hidden
competition correctness or a reliable population-level speed guarantee. The
families are distinct compositions of overlapping algorithmic concepts, not a
claim that every final concept was unknown during compiler development.

The organizer's runner has unlocked clocks. Balanced order and paired analysis
reduce some timing bias but cannot eliminate drift, contention or noisy tiny
kernels. Identical PTX across budgets can make the action space too small for an
informative learning result. The preregistered gate and negative outcomes must
remain visible; final feedback cannot be used to retune this same pilot and
retain a held-out claim.

Transport hashes are a reproducibility and accidental-corruption check, not an
authenticated server attestation. Individual HTTP attempts are retained, but a
timeout can reach the server without returning a response, so attempt counts
are not exact kernel-launch counts. The local runner lock does not exclude
other clients; the explicitly coordinated runner slot remains necessary.

These checks support a disciplined optimization experiment. They do not replace
compiler regression tests, independent GPU correctness checks, or careful
review of any later production change. No conclusion about RL superiority or
competition placement follows from this review.

## Reviewed revisions

SHA-256 values at the end of the source review:

| File | SHA-256 |
|---|---|
| `tools/selector_study.py` | `b495a2f0389463f8010a9a68be69f129cd8d8d786dce711c82bb73cf90626ba4` |
| `tests/test_selector_study.py` | `6cf67a13025f9bd4b75c5fb0b88ff4458d4c210e27270a2ae2f7796225b91a6c` |
| `tests/expressions_selector.py` | `0e48636e2411c0dce91111eb368c5c157e3d931f1b7d4e58d7e3916761852dc4` |
| `artifacts/validation/phase4/selector-preparation/product-oracle-repair.json` | `c5265d2e9f9acec66f96c0cfe9dad42bd6ebfe79a5b1d27c7df331305dc2f427` |

Later source or corpus changes require checking whether this review still
applies and regenerating/resealing the experiment as required by its protocol.

## Generation-compatibility and feasibility addendum

The expression source subsequently changed to SHA-256
`324ee3c50cd919efefe5856ddc2d184888e3c9ae090ade99bc950ae2e9c09980`.
The reviewer read the two narrow changes and the final-source evidence. The
selector tool and focused test source retain their hashes in the table above.

1. Pinned Emmy could not trace the direct `aten.prod` expression. The positive
   product case is now expressed as
   `exp(log(sigmoid(x * 0.125)).sum(-1))`; the independent NumPy reference still
   computes the direct product, with the same shapes and `rtol=1e-3`,
   `atol=1e-12`. This changes floating evaluation order and tests sum/log/exp
   lowering against a product oracle. It is **not native product-reduction
   lowering coverage**; separate typed product fixtures supply that coverage.
2. Score-bias attention declares and references inputs in `q, kt, bias, v`
   order to agree with generated graph inputs. Its mathematical operation,
   shapes and tolerances are unchanged. Seed assignment follows this new order,
   so the frozen explicit values, rather than old snapshots, define its cases.

The [final-source focused log](../artifacts/validation/phase4/selector-preparation/focused-tests-generation-repaired.txt)
records **22 passing tests**. The
[canonical preparation](../artifacts/validation/phase4/selector/preparation.json)
passes all 30 packages, including both product shapes under all three canonical
inputs; the unchanged `check_reference` guard rejects an entirely zero output
for each of those six cases. The
[final-source extra-seed audit](../artifacts/validation/phase4/selector-preparation/reference-final-spec.json)
passes **90 CPU oracle comparisons**, using seeds 37 and 89 normal and seed 137
wide. These are separate from the canonical inputs. Its source hash matches the
current specification and frozen copy. The earlier audit files and reviewed
source hashes remain historical evidence, not substitutes for these records.

The [static feasibility report](../artifacts/validation/phase4/selector/feasibility.json)
confirms **90/90 frozen-versus-production PTX comparisons**. Only one training
workload, zero validation workloads and one final workload have nonidentical
PTX across actions; each nonempty group covers only one family. Both held-out
splits fail the predeclared opportunity requirement. The proper outcome is to
retain fixed budget 1024 without fitting learners or collecting comparative
timings. This is a negative action-space result, not evidence that RL generally
cannot help compiler optimization.

The reviewer also read the original correctness-only driver, now retained at
`artifacts/validation/phase4/selector-correctness/driver-at-attempt.py`
(SHA-256 `1c4be18378005608707dcbb47812559b115b44730978e11e3b78e1f4efe6f386`).
It verifies the sealed 30-by-3 input roster and production emission identity,
runs only default PTX, checks every returned output, retains raw transport
artifacts, and rechecks source/seal stability. It does not fit policies or
aggregate performance. The owner's
[four passing mocked-driver tests](../artifacts/validation/phase4/selector-preparation/correctness-driver-tests.txt)
exercise successful checks, wrong outputs, wrong entry points, and changed
production source without reaching a network. These mock results and this
source review do not themselves establish any GPU execution result.

## Closed correctness-only GPU evidence

The later [fresh run](../artifacts/validation/phase4/selector-correctness-retry01/summary.json)
passes **90/90 checks across all 30 packages** with unchanged production and
sealed inputs. The [post-run audit](../artifacts/validation/phase4/selector-correctness-retry01/integrity-audit.json)
rehashes and reconstructs every compile/run request, returned output and HTTP
attempt: 120 received events, 90 passing output checks, no pending or unreferenced
events. The reviewer read the final reports and auditor source and verified
their reported hashes. There were no transport retries or summary-write retries
in the completed run. These results are additional correctness evidence; the
static rejection of the learning experiment remains unchanged.

A final read-only check independently confirmed that the report contains
exactly **30 rows and 30 unique IDs**, that the sealed roster also contains
30 unique IDs, and that the two ID sets are equal. Every row passes. This closes
the roster question for this artifact. The generic auditor currently uses row
and check counts for completion; explicitly enforcing roster equality there is
a future hardening improvement, not evidence missing from this reviewed result.

The [first attempt](../artifacts/validation/phase4/selector-correctness/integrity-audit.json)
remains incomplete after Windows denied an atomic summary-file replacement.
Its `.tmp` aggregate records 22 passing checks; the partial audit separately
validates one additional durable response. Neither is added to the fresh total.
The actual lock owner or cause was not established, and no kernel mismatch was
reported. Raw artifacts, the original driver and both summaries remain retained.

The reviewed retry driver has SHA-256
`512aa8c193021114d9388db9062492a372ff6f41e57aa0e6e8c77923b5618668`.
Its only execution-flow repair is a bounded retry of summary writes on
`PermissionError`; it does not replay a GPU request or alter checked values.
The [final seven passing mocked tests](../artifacts/validation/phase4/selector-preparation/correctness-driver-final-tests.txt)
include transient/persistent write failures and raw-response corruption. No
test-driven claim that the real retry path was exercised is made: its recorded
retry count was zero.

Closed evidence SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `selector-correctness-retry01/summary.json` | `a4c4c83b5f4f069481d28a41c4c6d586207c8675f3efd8e04d4b0b532f8c7ea0` |
| `selector-correctness-retry01/integrity-audit.json` | `c2a03dbbd32ba876b2b059ebd69250e3e7351ad2925faba9b74dbbbae9469b0e` |
