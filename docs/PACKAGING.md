# Reproducible submission packaging

`tools/package_submission.py` creates a minimal submission ZIP only after supplied
validation reports pass and identify the current source. It does not submit,
publish, install dependencies, assemble PTX, contact the organizer or execute a
GPU kernel.

The current runtime ZIP contains exactly these five files at its root:

- `compile.py`
- `ptx_backend.py`
- `ptx_tensor.py`
- `ptx_bulk.py`
- `requirements.txt`

Documentation, test inputs, generated PTX and validation evidence remain outside
the submission ZIP. The entry command after extraction remains
`python compile.py path/to/kernel.ir.json`, with that input's `meta.json` beside it.

## Current review/source bundle

The [phase-6 runtime pointer](../artifacts/release/phase6/latest.json) selects
the minimal compiler ZIP. The [review pointer](../artifacts/release/phase6/review/latest.json)
selects a separate bundle built by `tools/package_review.py`: compiler sources,
examples, tests, tools, documentation, required small fixture trees and explicitly
selected evidence. `REVIEW_MANIFEST.json` records every member hash. The review
bundle is extracted and its advertised feature tests and CLI commands are checked
before delivery. Neither packaging tool publishes the repository or submits an entry.

The [phase-6 release report](PHASE6.md) identifies the current evidence. The
older three-file archive described below remains unchanged and historical.

## Previous phase-4 evidence manifest (24 October 2026)

The [latest pointer](../artifacts/release/phase4/latest.json) now identifies
[manifest `3908fd...`](../artifacts/release/phase4/submission-bcbc4a8381497f1d34be-3908fd80497591da4d12.manifest.json),
SHA-256 `3908fd80497591da4d12af75b4203bbffeceb5c3372d1a046e02450a217a95ec`.
It retains six exact evidence snapshots and a fresh extracted-CLI check.
Production source did not change, so it references the **same submission ZIP**,
SHA-256 `b9af6a761233e39b8faa330e1fd01c49e8a457498cb5d3b7e2465ef5403251e8`.
The original manifest and snapshots remain intact.

The added [24 October review](FINAL_CHECK_2026_10_24.md) records a fresh 15/15
public-client pass plus extra shape/type/operation checks. Its full mathematical
accounting is **70 passes and 9 retained failures out of 79 executions**. Those
nine outputs match explicit FP32 IR semantics exactly but cannot meet the
original FP64 mathematical precision contract. Separate FP64-state **test IR
variants** pass the unchanged references and tolerances. Neither the packaging
success nor the review's acceptance flag relabels the original nine failures.
See the [closed review index](../artifacts/validation/2026-10-24-final/summary.json),
including `original_extra_math_gate_pass: false`, for the release criterion and
source-linked evidence. This is not a claim that every generated case passed.

The new extracted CLI check emits the same 1,524-byte public add PTX, SHA-256
`c0d81f88369a245e4144ff5e364ed1139349e24a40f8b3ca26f6257b41729a41`,
with matching extracted source bytes and no audit-hook blocked events. No GPU
launch occurs during packaging; the numerical results come from the linked
runner reports.

## Creating the release

Run from the repository root after the final gates complete. Provide the actual
passing JSON report paths, not their logs or an intermediate in-progress report:

```console
python tools/package_submission.py --release-dir releases --evidence <final-numerical-summary.json> <final-cpu-report.json> <clean-install-report.json> --update-latest
```

Angle-bracket names above are placeholders for the finalized reports. `--evidence`
accepts one or more paths and may be repeated. `--update-latest` is optional; it
updates only the small `latest.json` pointer. Existing versioned archives,
manifests and evidence snapshots are preserved.

Gate acceptance is deliberately strict:

1. Each report must be an unambiguous JSON object with literal top-level
   `"pass": true` or `"ok": true`. Numeric `1`, string `"true"`, missing status,
   duplicate JSON keys, conflicting status flags and explicit failed subordinate
   `pass`/`ok` flags are rejected. Top-level errors or unsuccessful exit codes also
   reject the report. A child diagnostic can correctly expect a nonzero exit code
   while its own `pass` flag remains true.
2. Each report must contain a recognized production source-hash mapping. Accepted
   fields are `source_sha256`, `sources_sha256`, `compiler_sources_sha256`,
   `sources_before_sha256`, `sources_after_sha256`,
   `compiler_sources_before_sha256`, `compiler_sources_after_sha256`, and
   mapping-valued `compiler_sha256`.
3. Every reported hash for the five runtime files
   must match the current file, including both before/after mappings when present.
   Windows and POSIX paths are matched by basename so an immutable candidate copy
   can identify the same bytes. Every report must identify at least one production
   file; the supplied reports together must identify all five.
4. CPU reports may also identify tests and tools. Those extra identities are
   retained in the original report but are not interpreted as production files.
   Nested experimental-job compiler hashes may refer to tested wrappers; the
   summary's top-level mappings identify the production candidate.
5. Sources and supplied reports are checked again after the extracted CLI run.
   A change during packaging aborts publication.

The tool does not infer which gates constitute a complete release policy. Supply
the finalized numerical, CPU/assembly and clean-install gates required by the
project's release checklist. A successful report is evidence about its documented
scope, not hidden-set correctness or a guarantee of winning.

## Artifact identity and retention

The release directory receives:

| Artifact | Purpose |
|---|---|
| `submission-<source-id>.zip` | Five runtime files with deterministic ordering, fixed 1980 timestamps, fixed file modes and stored compression |
| `submission-<source-id>-<manifest-id>.manifest.json` | Source hashes, ZIP hash, selected evidence identities/statuses and extracted-CLI verification |
| Matching `.manifest.json.sha256` | SHA-256 of the complete manifest bytes |
| `evidence/<full-report-sha256>.json` | Exact immutable snapshots of the supplied reports |
| `latest.json`, if requested | ZIP and manifest paths plus both hashes |

The source identity hashes the canonical map of the five runtime-file hashes.
The manifest identity hashes the complete manifest. The full digests are retained
inside the manifest or its checksum, even though filenames abbreviate identities.
A filename collision with different bytes fails; it never replaces an existing
artifact. Repeating the same package with the same sources, evidence paths and
bytes, tool version and Python version produces the same ZIP and manifest bytes.
The ZIP itself depends only on the five captured runtime files and fixed archive
metadata; no creation clock or compression-library variability is introduced.

Different passing evidence can produce a new manifest referencing the same ZIP.
The exact original report paths and hashes are recorded alongside retained
content-addressed copies. These hashes detect accidental changes; they are not
cryptographic signatures or an independent attestation of report truthfulness.

## Extracted CLI verification

Before writing the release, the tool builds the ZIP in memory, checks the exact
member inventory and CRCs, extracts it into a temporary directory and compares
every extracted source byte with the captured original. A separate temporary
input directory receives the real public `p00_scalar_add` IR and its metadata.

It then invokes the extracted `compile.py` with the current Python interpreter.
The test requires a zero exit code, empty stderr, ASCII PTX beginning with
`.version`, and the entry name prescribed by the copied metadata. It verifies
that the package files and inventory remain unchanged. The manifest records the
input, metadata and output PTX hashes, output size and source identities.

A temporary Python startup audit hook rejects socket operations, child process
creation, known CUDA library loads and imports of external GPU/compiler
frameworks. The audit must finish with no blocked events. This is regression
instrumentation for the known compiler, not an operating-system sandbox against
arbitrary native code. Neither the audit helper nor test input enters the ZIP.

This smoke check uses already installed dependencies and does not replace the
separate clean-install gate. Emmy remains pinned by `requirements.txt`; NumPy,
requests and transitive dependencies are not hash-locked by this packaging step.

## Original verified Phase 4 release (23 October 2026)

The frozen backend is
`5cb3a351840ca03055b9725471fd3b44afee7a1d8e2832c49d439018da5ba9fc`.
Its release was created with this actual command from the repository root:

```powershell
python tools/package_submission.py --release-dir artifacts/release/phase4 --evidence artifacts/validation/phase4/release-remote/summary.json artifacts/validation/phase4/semantic-final/summary.json artifacts/validation/phase4/cpu-regressions.json artifacts/validation/phase4/clean-install/summary.json artifacts/validation/phase4/public-ptx-identity.json --update-latest
```

The immutable [submission ZIP](../artifacts/release/phase4/submission-bcbc4a8381497f1d34be.zip)
has SHA-256
`b9af6a761233e39b8faa330e1fd01c49e8a457498cb5d3b7e2465ef5403251e8`.
Its [manifest](../artifacts/release/phase4/submission-bcbc4a8381497f1d34be-b88459958203b4fdede2.manifest.json)
has SHA-256
`b88459958203b4fdede27e69a685283b4b67bfe26c20f9e7432c40a0479f486a`.
The pointer identified that pair when this release was created. It now selects
the 24 October evidence manifest above; this original manifest is preserved.

All three ZIP member bytes match the frozen sources, and all five retained
evidence snapshots match their original report bytes and hashes. The gate scope
is **236 default and 6 variant numerical checks across 120 distinct workload
packages**, **55 passed CPU test methods with 18 explicit GPU skips**, **21
successful clean-install CLI invocations across seven packages plus three
expected diagnostics**, and **15 public PTX identity checks**. The
[aggregate release summary](../artifacts/validation/phase4/release-summary.json)
links the original evidence with full hashes; [Phase 4](PHASE4.md) explains the
changes and limits. Repeated seeds and compiler variants are checks, not new
workload packages.

The [post-package verification](../artifacts/release/phase4/post-package-verification-b9af6a761233e39b8faa.json)
independently repeated the real extracted CLI invocation. Its stdout was
byte-identical to
`build/open-ptx-phase4-cli/p00_scalar_add/k_add_pointwise.ptx`: 1,524 bytes with
SHA-256 `c0d81f88369a245e4144ff5e364ed1139349e24a40f8b3ca26f6257b41729a41`.
The final organizer-run report records the same submitted PTX hash and size.
That report contains payload identities rather than raw PTX, so this last link
is hash-and-size verification, not another direct byte comparison. The
post-package verification report itself has SHA-256
`f6b2450b8432c4ca132ee734f329604cf0e5d05014c7826ef201ee85a09e57f0`.

For a concrete input after extraction, run the following from a directory that
also contains the public problem tree, or substitute the absolute path to that
actual input:

```powershell
cmd /c "python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx"
```

The problem tree is included in the review bundle, but not in a minimal runtime ZIP. Preserve the
input's companion `meta.json` when copying an IR into another directory.
