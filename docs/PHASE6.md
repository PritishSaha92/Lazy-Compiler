# Final research-informed release

This release follows the complete review of the locally present DeepInfer
checkout and the Mixture of Kittens article. The [decision record](RESEARCH_REVIEW.md)
and [source coverage manifest](research/source-review-manifest.json) explain
what was read, what transferred, and which ideas require a different execution
contract. This is the current release; [phase 5](PHASE5.md) remains historical.

## Compiler and test changes

The only production-code change is a nine-line constructor check in
[ptx_backend.py](../ptx_backend.py). Native `CpAsyncCopy`, `CpAsyncCommit` and
`CpAsyncWait` now reject an explicit target below SM80 before emitting PTX.
Nested operations are included. The explicitly requested synchronous-copy
implementation remains available. The previous implementation's invalid SM75
emission and assembler diagnostics are preserved in the
[pre-fix record](../artifacts/validation/phase6/async-target-before.json).
Six [architecture tests](../tests/test_async_target.py) cover this correction.

Three [new ring examples](../examples/ring) combine two shared slots, native
bulk copies, shared matrix loads and tensor MMA across five or seven K16 slabs.
They cover repeated parity transitions, prefetch, final drain and disjoint CTA
outputs. The BF16 case uses F32 wire buffers and explicit scheduled conversion
to BF16. Independent exact references and zero tolerances detect missing or
stale slabs. A same-process compilation test alternates targets and numerical
modes, then verifies earlier PTX and input bytes remain unchanged. The four
methods live in [test_ring_pipeline.py](../tests/test_ring_pipeline.py).

There is no new distributed runtime, rescheduling, quantization mode, learned
policy or performance claim. All five runtime files retain the existing direct
IR-to-PTX translation and prescribed launch/ABI contract.

## Validation on the final runtime sources

| Gate | Result | Exact evidence |
|---|---:|---|
| Unmodified organizer client | 15/15 public problems passed | [Report](../artifacts/validation/phase6/public-submit/summary.json) |
| Organizer GPU feature corpus | 29/29 packages passed | [Report](../artifacts/validation/phase6/feature-runner/summary.json) |
| Independent raw-exchange audit | 29 comparisons replayed; 58 events and 58 HTTP attempts verified | [Audit](../artifacts/validation/phase6/feature-runner/integrity-audit.json) |
| Complete CPU/assembler discovery | 99 passed, 18 explicit local-GPU skips; 117 methods total | [Report](../artifacts/validation/phase6/cpu-regressions.json) |
| Isolated installed-dependency audit | 14 packages; 42 successful compilations and three expected diagnostics | [Report](../artifacts/validation/phase6/clean-install/summary.json) |
| Frozen baseline comparison | 120/120 retained scalar modules emit identical PTX | [Report](../artifacts/validation/phase6/ptx-compatibility.json) |
| Previous feature-module comparison | All 26 previous feature PTX hashes unchanged | [Release index](../artifacts/validation/phase6/release-summary.json) |

This is **44 successful organizer-GPU checks** in this release: 15 public and
29 additional feature packages. The latter comprise 17 tensor examples, six
bulk examples, two earlier combined pipelines, one unchanged Emmy-generated
ragged matmul and three new ring pipelines. The
[input manifest](../examples/research-feature-manifest.json) pins their exact
IR, metadata and independent expected arrays. No local GPU was used.

The [release index](../artifacts/validation/phase6/release-summary.json) verifies
source hashes, input stability, public-client logs, independent audit identity,
clean-install results and the unchanged prior PTX. These numerical executions
do not imply sanitizer coverage, a hidden-set ranking, or a speed advantage.
The runner's one timed iteration remains indicative feedback.

## Runtime identity

| File | SHA-256 |
|---|---|
| `compile.py` | `4ed3fb67833b7de20693c7cb3f163a4646cac8d3a71e63649c1af1bccd37ce56` |
| `ptx_backend.py` | `c718b8d2ec156626e4c74eb246632b0d430235bcf11f25e2643b4e8c185c6df6` |
| `ptx_tensor.py` | `5d97c908dc0f397401cbe7894343dd1bbc3a2d8d3b0827fe650ccb3a3015e935` |
| `ptx_bulk.py` | `ab63f516681e78e06f4b7403b24018c60c8119b8c92c34f00236d74806406c65` |
| `requirements.txt` | `5d054f8757b1212865653105a4e7c634e2952007da020d2f67ad456eb6bccc84` |

## Artifacts and submission

The [runtime release pointer](../artifacts/release/phase6/latest.json) selects
the five-file compiler ZIP. The [review/source pointer](../artifacts/release/phase6/review/latest.json)
selects the complete project with examples, tests, documentation and evidence.
Use the extracted review/source bundle for the public code repository, with
`compile.py` at its root. Distribution pointers refer to the repository release
layout; an archive cannot embed its own final pointer. The extracted bundle's
`REVIEW_MANIFEST.json` identifies every included member.

Use the current [submission overview](../SUBMISSION.md),
[form answers](SUBMISSION_FORM.md), and [609-word video script](VIDEO_SCRIPT.md).
Packaging does not publish a repository, upload a video or submit an entry.

The nine earlier generated FP64-buffer/FP32-state mathematical-reference
failures remain explicitly recorded in the [earlier review](FINAL_CHECK_2026_10_24.md).
Their separate true-FP64 variants and exact IR-schedule attribution are not
relabelled as fixes to the original mathematical comparisons.
