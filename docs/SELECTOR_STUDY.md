# Bounded unroll selector study

This is a development experiment, separate from the submitted compiler. Its purpose
is to determine whether structural IR features help choose among three legal
unrolling budgets on unseen expression/schedule families. No claim of improvement,
RL superiority, or hidden-set coverage follows from preparing the experiment.

The completed static audit rejected this pilot's action space before comparative
GPU timing: only one training workload, no validation workloads, and one final
workload produce different PTX across the three budgets. Both held-out splits fail
the preregistered minimum of six. **Keep fixed budget 1024.** No policy was fitted,
no validation winner was selected, and no runtime speedup is claimed. The authored
30-workload corpus remains useful for independent compiler correctness checks.

| Split | Workloads | Workloads with distinct PTX choices | Distinct families | Unmarked plain loops |
|---|---:|---:|---:|---:|
| Train | 10 | 1 | 1 | 5 |
| Validation | 10 | 0 | 0 | 7 |
| Final | 10 | 1 | 1 | 5 |

The 90 configurations deduplicated to 32 workload/configuration modules, all of
which passed CPU assembly. All 90 PTX outputs also matched the final production
backend `5cb3a351840ca03055b9725471fd3b44afee7a1d8e2832c49d439018da5ba9fc`
byte for byte. Evidence is in
[`feasibility.json`](../artifacts/validation/phase4/selector/feasibility.json).
Unmarked loops were counted for diagnosis; their schedules were not changed to
manufacture a positive result in this study.

## Frozen scope and corpus

The only action is `compile_file(ir, unroll_budget=0|256|1024)` against the frozen
historical baseline backend (`8afa55d...`). The completed 90-configuration identity
audit bridges those outputs to the final production backend (`5cb3a351...`). All
remaining numerical defaults, the global expansion
cap of 4096, launch geometry, memory effects and synchronization stay fixed. The
baseline is the existing budget 1024. No arithmetic flags are learning actions.

[`tests/expressions_selector.py`](../tests/expressions_selector.py) declares 30
workloads: 15 expression/schedule families, two sizes per family, ten workloads and
five categories in each split. Shapes, knobs and memberships are fixed before any
timing. Different seeds of a workload never cross splits.

| Split | Pointwise | Reduction | Matmul | Fused | Attention |
|---|---|---|---|---|---|
| Train | Broadcast affine-ReLU | Row inner product | Plain GEMM | Weighted RMSNorm | Scaled attention |
| Validation | Rational sigmoid gate | Column maximum | GEMM+bias+ReLU | Softmax-weighted sum | Attention+residual |
| Final | Three-dimensional absolute difference | Log-domain sigmoid product | Batched GEMM | Centered column L2 normalization | Score-bias attention |

This is a small feasibility pilot. Some concepts appeared in earlier development
tests; the claim is a frozen family split within this study, not that the author has
never seen these algorithms. There are only five final families, so uncertainty can
remain large even when there are ten shapes.

## Preparation and sealing

`tools/selector_study.py prepare` performs CPU work only:

1. Validate family separation and record a draft protocol/roster.
2. Check independently implemented NumPy references against CPU PyTorch for each
   expression, using normal seeds 0/17 and scaled seed 101. Inputs are finite FP32;
   scaled inputs use a factor of eight. No tolerance is changed after timing.
3. Require the installed `emmy-ml==0.3.11` distribution. Generate valid scheduled IR
   and exact launch metadata, or verify an existing package's expression, tolerance,
   generation knobs, source/generator hashes and every packaged artifact. Stale or
   unproven packages are regenerated. Require single-kernel eligibility.
4. Extract structural numeric features with `tools/analyze_ir.py`. Names, filenames,
   categories, seeds and input/output values do not enter the models.
5. Freeze explicit input values and independently computed expected outputs. Verify
   the generator's baseline `expected.npz` matches the frozen oracle exactly.
6. Emit all three actions with the frozen backend, record compilation time/size,
   deduplicate identical PTX by SHA256, and use local `ptxas -arch=sm_89` for CPU
   assembly checks. This does not run a local GPU. An explicit `--skip-assembly`
   leaves that check omitted in the preparation records.
7. Seal every relevant source, IR, metadata, expected-output/input snapshot, feature
   report and PTX hash. Any later mutation invalidates the experiment.

Generation/oracle repairs are permitted only before sealing and timing. A changed
draft requires a recorded `--repair-note`; previous draft and diagnostics remain
visible. Failures are retained. A failed package is not quietly removed or replaced
after performance feedback. The preparation report must pass for all 30 workloads.

An additional independent CPU sanity pass used seeds 37/89 and scaled 137; all
90 checks passed on the initial draft before generation. Independent review then
found that the original sigmoid product could accept an all-zero implementation
under a generic absolute tolerance. Before sealing or timing, its logits were
scaled by 0.125 and its absolute tolerance tightened to 1e-12. The original source
and reason remain in `generation-history/` and `preseal-repairs.json`. Both repaired
shapes pass all three canonical CPU checks and explicitly reject all-zero output;
the six-row evidence is `../artifacts/validation/phase4/selector-preparation/product-oracle-repair.json`.
These are oracle-validation results, not GPU correctness evidence. Canonical seeds
remain 0/17/101. The focused protocol suite passed 22 tests; its log is alongside
that product report as `focused-tests.txt`.
A fresh 90-check pass on the repaired source, again using normal seeds 37/89 and
scaled 137, is retained separately as `reference-sanity-repaired.json`; all 90
checks passed. The earlier draft's report remains available for provenance.

CPU generation then exposed two additional preparation issues, before any study
GPU evaluation. Emmy 0.3.11 could not trace `aten.prod` as a dimension reduction,
so the same positive sigmoid-product function is expressed as
`exp(log(sigmoid(x * 0.125)).sum(-1))`. Its NumPy oracle still computes a direct
product, and its tolerance is unchanged. This generated case exercises sum/log/exp
lowering, **not native product-reduction lowering**; direct typed product fixtures
cover that operation elsewhere. Score-bias attention also required matching input
declarations and reference parameters to Emmy's graph order `q, kt, bias, v`.
The mathematical function, shapes, tolerances and family split did not change.

The initial generation's 26 successes and four failures remain recorded. Two
product probes and two attention probes generated valid single kernels, each
retaining its actual source bytes and per-spec provenance; the other 26 packages
were reused only after exact signature and artifact verification. The final
30-package preparation passed all 90 canonical CPU reference checks and all 90
extra-seed checks in `reference-final-spec.json`. The final focused suite passed
22 tests (`focused-tests-generation-repaired.txt`). The separate correctness-only
driver was checked with network-free mocks covering the full 90-case roster,
output/entry failures, production-source mutation, retained transport hashes and
bounded evidence-write retries. Its completed runs and audit evidence are described
below.

## Completed correctness-only evaluation

After the static rejection, the default budget 1024 was tested on all 30 sealed
workloads using normal seeds 0 and 17 and scaled seed 101. The fresh completed run
passed **90 of 90 independent-output checks** on the organizer's GPU. It made 30
assembly requests and 90 run requests, with 120 received HTTP responses, no HTTP
retry and no evidence-write retry. The final production compiler hashes stayed
unchanged. Each run used zero warmups and one iteration; its raw timing field is
retained as transport evidence and is not used to compare policies or claim speed.

The closed report is
[`selector-correctness-retry01/summary.json`](../artifacts/validation/phase4/selector-correctness-retry01/summary.json).
The separate offline
[`integrity-audit.json`](../artifacts/validation/phase4/selector-correctness-retry01/integrity-audit.json)
passed after rehashing every request, response and HTTP-attempt artifact,
reconstructing all 90 frozen inputs and launch contracts, and independently
rechecking every returned output. It also confirmed chronological event coverage,
the complete 30-workload roster, unchanged source/seal hashes and no orphan or
pending responses. The report and audit SHA-256 values are respectively
`a4c4c83b5f4f069481d28a41c4c6d586207c8675f3efd8e04d4b0b532f8c7ea0`
and `c2a03dbbd32ba876b2b059ebd69250e3e7351ad2925faba9b74dbbbae9469b0e`.
The 90 checks are additional correctness evidence, not a successful selector
experiment. Fixed budget 1024 remains in production.

An earlier attempt stopped because Windows rejected replacement of the evidence
summary file with `PermissionError`. The exact cause of that file-access failure
is unknown. Its original driver, summary, temporary summary and raw responses are
preserved in
[`selector-correctness/`](../artifacts/validation/phase4/selector-correctness/).
That attempt recorded 22 passing checks; one further successful raw run response
was retained before the failed summary write and independently validated during
the forensic audit. **All 23 earlier returned run responses are excluded from the
fresh 90-check total.** No failed kernel output prompted the retry. Only the private
evidence writer gained bounded retries for `PermissionError`; the compiler,
sealed corpus, oracle and launch contract were unchanged.
The earlier attempt's separate
[`integrity-audit.json`](../artifacts/validation/phase4/selector-correctness/integrity-audit.json)
records intact retained evidence while explicitly marking that attempt as an
incomplete 90-check gate.

The final private driver/auditor regression suite passed **seven network-free
tests**. In addition to output, entry-point, source-change and writer-retry checks,
the last test audits a complete mocked 90-check report and then confirms that
changing one retained raw response fails hash verification. Its result is in
[`correctness-driver-final-tests.txt`](../artifacts/validation/phase4/selector-preparation/correctness-driver-final-tests.txt).
Six driver tests had already passed before the fresh GPU run; the seventh test
checks the separate offline auditor and did not change the running driver.

## Serialized remote measurement

This section describes the implemented comparative protocol. **It was not run for
this pilot**, because the static opportunity gate failed. The separate default-only
correctness checks below do not train a selector or compare performance.

Only an explicitly assigned exclusive runner turn may invoke `run`. CPU preparation
does not contact the service. The implementation reuses the existing official
`remote_post` wrapper and records the turn label, server and complete request and
response artifacts. A study-local exclusive lock prevents duplicate clients for
this study; coordination with other studies still requires the runner turn.

For each workload, identical PTX actions share one assembled module and one set of
measurements. The 1024 action is the canonical representative when it aliases another
action. No fake speedup is inferred by timing identical modules independently.

Every distinct candidate first passes all three independent input checks. Timing is
allowed only when every candidate for that workload passes. Six rounds use all six
permutations for three distinct candidates, alternating AB/BA for two, and six
baseline samples for one. Each request uses three warmups and twenty iterations.
Every returned output is checked against the independent oracle; missing outputs,
shape errors, truncation, nonfinite values and invalid timings fail the request.
This checks the output returned by a repeated-run request, not each internal launch.

All timing calls receive the same explicitly stored input values. Output zeroing
and argument order follow the launch contract. Candidate assembly responses,
resources, compiler time, code size, all failures and each measurement remain in
durable reports. Transport failures are distinct from kernel/output failures. The
existing wrapper's bounded infrastructure/busy retries are retained individually
with HTTP status, response body, elapsed time and hashes; their actual attempt count
is reported separately from returned timing labels. A timed-out request may have
reached the server, so attempts are not assumed to equal successful kernel runs.
The tool does not overwrite a failed split's measurements. There is no parallel
runner queue and no local GPU execution.

Before fitting or evaluating a policy, the reader rehashes every raw request,
response and HTTP-attempt artifact. It reconstructs the sealed PTX alias map,
requires the assembled entry point, matches exact inputs/launch parameters/cubin
identities, reruns independent output validation, and verifies that actual event
numbers follow the balanced measurement chronology. Altered summary timing fields
cannot substitute for retained output evidence.

## Policies and equal feedback budgets

The implementation can compare four policies; none was fitted or compared using
this pilot's GPU measurements:

- **Fixed 1024:** current production policy, zero learning feedback.
- **Fixed heuristic:** use 1024 when `hinted_loop_work_max <= 256`, 256 when it is
  at most 1024, and zero above that; missing information falls back to 1024.
- **Shallow tree:** seeded deterministic regression splits, depth at most two,
  at least two workloads per leaf and two observations to support a nonbaseline
  leaf action. Training uses a seeded balanced logger for alternatives 0/256.
- **Contextual bandit comparator:** separate ridge-linear reward estimates with
  LinUCB exploration (`alpha=0.5`, ridge 1), selecting one alternative 0/256 for
  each training context. Validation/final decisions use frozen predicted reward,
  without exploration or updates, with baseline-winning ties.

In an eligible future study, both learners would receive exactly **20 logical
aggregate reward observations**: one
baseline and one alternative for each of ten training workloads. Reward is the
median paired log speedup over baseline. The exact accesses and distinct physical
labels are recorded, so PTX aliases do not masquerade as independent observations.
Only training features determine feature columns, centering and scaling; missing
columns use zero. Feature values receive a signed log1p transform.

That protocol measures all three actions once to support held-out comparison and an
oracle ceiling. The learner functions expose only their recorded training labels.
Their adaptation is **offline replay of a previously measured table**, not fresh
on-policy GPU evaluation, not online RL training, and not extra independent samples
when a label is read again. Fixed policies cost no training feedback; their smaller
cost remains visible. The measured oracle is an optimistic diagnostic, not a policy
that may inspect final answers at compilation time.

## Validation choice, final embargo and promotion

These are predeclared rules for an eligible comparative study, not stages executed
in this statically rejected pilot.

The tree and bandit hyperparameters above are predeclared. Validation compares the
fixed heuristic, tree and bandit and freezes the highest-gain qualifying policy;
otherwise it freezes fixed 1024. `selection.json` hashes the models and training/
validation evidence. Final measurements cannot start before this record exists.
After final results exist, refitting and reselection are prohibited, including when
the results are disappointing.

Uncertainty resamples expression families and then paired timing rounds within each
workload, using 4,000 seeded bootstrap draws. Report the geometric mean of per-workload
median paired speedups, the 95% interval, worst workload and every individual result.
Do not treat six repeated timings or two shapes from one family as independent
algorithm families.

A policy qualifies only when **all** these conditions hold:

- Every candidate output is correct and all frozen evidence is intact.
- At least six workloads from at least two families have nonidentical PTX choices.
- Geometric-mean gain is at least 3%, and the paired uncertainty interval lies
  strictly above a speedup of 1.0.
- No workload's median paired slowdown exceeds 5%.

The same gate applies on validation and then the untouched final split. A failed
final gate keeps fixed 1024; another final-scoring policy cannot replace the frozen
selection. Too few choices or noise-dominated results mean inconclusive/no promotion.
Even a pass produces an eligibility report for a separate review, never an automatic
production edit. Compilation cost, assembly resources, PTX size, feedback budget and
worst regressions accompany any claimed runtime gain.

## Commands

These commands document reproduction on a **new** output directory. The actual
sealed pilot is retained and must not be overwritten. From the repository root:

```powershell
python tools/selector_study.py prepare --out artifacts/validation/phase4/selector-new
python -m pytest tests/test_selector_study.py
```

Only if a separately frozen study passes its static opportunity gate and an
exclusive remote turn is assigned, its prospective staged sequence is:

```powershell
python tools/selector_study.py run --out artifacts/validation/phase4/selector-new --split train --runner-turn ASSIGNED_TURN
python tools/selector_study.py run --out artifacts/validation/phase4/selector-new --split validation --runner-turn ASSIGNED_TURN
python tools/selector_study.py fit --out artifacts/validation/phase4/selector-new
python tools/selector_study.py run --out artifacts/validation/phase4/selector-new --split final --runner-turn ASSIGNED_TURN
python tools/selector_study.py evaluate --out artifacts/validation/phase4/selector-new
```

`artifacts/validation/phase4/selector/` contains the draft, preparation diagnostics,
sealed corpus, input/oracle snapshots, unique PTX, static analyses, generation
history and the negative feasibility report. It contains no fitted models,
learning-feedback ledger, validation selection or comparative timing results.
Those artifacts belong only to the prospective stages described above.
The submitted `compile.py` and `ptx_backend.py` are not changed by any study command.
