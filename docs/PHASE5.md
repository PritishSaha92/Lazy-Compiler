# Optional hardware features and manual-review release

This is the previous hardware-feature release. The current release is [PHASE6.md](PHASE6.md).

On 24 October 2026, the user relayed organizer guidance that the public set is
a baseline, submissions receive manual review, and extra tensor-core, copy-engine
and datatype support should be demonstrated with examples and tests. This release
adds those reviewable extensions while preserving the scalar backend's behavior.

## What changed

`ptx_backend.py` delegates explicit tensor and bulk node types to two small
modules. [ptx_tensor.py](../ptx_tensor.py) implements the m16n8k16 FP16/BF16-to-FP32
path, native shared fragment loads, guarded global gathers, fragment families,
repacking and scalar epilogues. [ptx_bulk.py](../ptx_bulk.py) implements aligned
in-bounds rectangular bulk transfers with mbarrier arrival and parity protocols.
The CLI selects SM90 for explicit bulk/barrier nodes and retains SM89 otherwise.
Explicit API targets retain their requested architecture.

Neither extension performs general tensorization or rescheduling. The bulk path
does not require a host tensor-map argument and is not general TMA tensor-map
lowering. [FEATURES.md](FEATURES.md) documents the exact supported forms and
rejections. The ordinary scalar numerical policy is unchanged.

Independent review found two new alignment-proof edge cases before GPU testing:
a repeated `LdmatrixLoad` object needed the intersection of all surrounding loop
facts, and a late loop-carrier mutation invalidated the next iteration's alignment
assumption. Both received negative regressions. Auxiliary-thread Tile partitions
must also preserve complete warps. Prior private preparation exports are retained
as intermediate evidence; the final run uses the later frozen sources.

## Validation

| Gate | Result | Evidence |
|---|---:|---|
| Unmodified public client | 15/15 numerical passes | [Report](../artifacts/validation/phase5/public-submit/summary.json) |
| Optional features on organizer GPU | 26/26 numerical passes | [Report](../artifacts/validation/phase5/feature-runner/summary.json) |
| CPU and assembler regression discovery | 89 passed, 18 explicit local-GPU skips; 107 methods total | [Report](../artifacts/validation/phase5/cpu-regressions.json) |
| Isolated installed-dependency audit | 11 packages, 33 successful compilations, three expected diagnostics | [Report](../artifacts/validation/phase5/clean-install/summary.json) |
| Frozen scalar baseline compatibility | 120/120 input modules emit identical PTX | [Report](../artifacts/validation/phase5/ptx-compatibility.json) |

Thus this release has **41 successful organizer-GPU executions**: 15 public and
26 optional-feature checks. These counts do not add older repeated public runs
or offline comparisons as new GPU coverage. No local GPU was used. The complete
[release index](../artifacts/validation/phase5/release-summary.json) records the
source identity, evidence hashes and raw-data audit.

The 26 feature packages comprise 17 manually constructed tensor examples, six
bulk examples, two combined pipelines and one unchanged Emmy-generated tensor
matmul. [The manifest](../examples/feature-manifest.json) hashes the input corpus.
The [original generated package](../examples/generated_tensor/provenance.json)
retains its IR, metadata, tolerances and expected arrays byte-for-byte. It is the
previously rejected 11×7 by 7×19 FP16 case; it now executes through the tensor
lowerer. The combined pipelines use exact dyadic inputs and zero tolerance, so
discarding a slab or resetting the accumulator cannot pass by numerical slack.

The six bulk tests include multi-CTA offsets, rectangular and rank-five crops,
multiple arrivals, repeated phases, a two-slot ring, and genuine FP64 storage
whose payloads cannot survive an FP32 round trip. Tensor tests include FP16/BF16,
ragged M/N/K, paired and transposed B, shared swizzles, global gathers, indexed
families, asynchronous staging, guarded fusion, atomics and repacking.

Raw feature requests, decoded responses and individual HTTP attempts are
retained. The independent audit checks request ABI, input bytes, expected arrays,
output dtype/shape, PTX/cubin linkage, source hashes and the exact planned roster.
One timed iteration per canonical check is indicative runner feedback; there is
no comparative performance claim.

## Evidence boundaries and artifacts

The [24 October scalar review](FINAL_CHECK_2026_10_24.md) still records nine
failed mathematical comparisons where generated FP64 buffers carried FP32
arithmetic state. Their exact IR-schedule matches and separate true-FP64 test
variants remain as originally recorded. Adding hardware features does not make
those original mathematical-reference failures disappear.

The runtime [release pointer](../artifacts/release/phase5/latest.json) selects
the five-file compiler ZIP and source-bound evidence manifest. The separate
[review pointer](../artifacts/release/phase5/review/latest.json) selects the
source bundle with examples and tests. Its manifest hashes every member.
[PACKAGING.md](PACKAGING.md) describes the extracted-CLI and archive checks.

The existing phase-4 release and its reports remain historical artifacts.
Use [SUBMISSION.md](../SUBMISSION.md) for the current reviewer-facing overview.

The final runtime archive is `submission-a762243a66b4b7586ac6.zip`, SHA-256
`598fe981e4471f46a3417cf2760412ee3fa6e39e66289c4beef425a30ac1c116`.
Its extracted CLI check passed, with all five source files matching the release
manifest. The review bundle has its own manifest because it also contains tests
and documentation.

The review pointer above belongs to the repository release layout. Inside an extracted review bundle, use REVIEW_MANIFEST.json for its exact inventory; the archive cannot embed its own final release pointer.
