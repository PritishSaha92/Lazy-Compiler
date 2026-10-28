# Source checkout

This repository contains the current phase-9 compiler, examples, tests and
technical documentation. Use this checkout for the public repository; a new
ZIP is not required. See [PHASE9.md](PHASE9.md) for changes and validation status.
The phase-9 public, feature and CPU gates passed; the separate generated
composition study retains its 21/24 mathematical result. The
[current verification report](../artifacts/validation/phase9/checkout/verification.json)
records the separate inventory, documentation-link and CLI checks. The earlier
[phase-8 release summary](../artifacts/validation/phase8/release-summary.json)
remains evidence for its own source revision.

The source checkout omits presentation drafts and the scratch `out.ptx` file.
Python caches and local compiler outputs are ignored. Development and code
review were AI-assisted. Git metadata is not part of the checkout inventory.

`.gitattributes` disables automatic newline conversion so Git preserves the
source and evidence hashes on both Windows and POSIX checkouts.

`REVIEW_MANIFEST.json` uses the version-2 checkout schema. It records the current
file inventory and runtime hashes, excluding Git metadata, caches, local scratch
files and checkout verification reports. Its `current_validation` references
bind the current runtime to its release report and 20 CLI expectations: sixteen
new-feature cases (eight GELU and eight fragment packages) plus four retained cases.
It is not the inventory of the earlier ZIP. The original
[archive inventory](../artifacts/validation/phase7/source-archive-inventory.json)
and [extraction report](../artifacts/validation/phase7/review-bundle-verification.json)
remain available as provenance. Their runtime hashes describe phase 7, not the
changed phase-9 backend. Historical validation reports retain their original
paths, hashes and outcomes; the verifier checks that provenance separately.

Run these commands from the repository root:

```console
python tools/check_checkout.py --report artifacts/validation/phase9/checkout/my-verification.json
python tools/run_regressions.py --report artifacts/validation/phase9/checkout/my-cpu-regressions.json
```

The release check uses
`artifacts/validation/phase9/checkout/verification.json`; the commands above use
fresh reviewer report names in the same excluded report directory. It verifies
source hashes, documentation links and actual CLI output for 20 examples.
Uniform LF output may be compared with retained Windows CRLF after that explicit
conversion; all other bytes must match. Use a new `--report` path when repeating
a check, because previous reports are not overwritten. The older
`artifacts/validation/checkout-current/` reports remain historical evidence.
The regression suite uses `ptxas` for
assembly checks when installed; local-GPU tests remain opt-in. These commands
do not contact the organizers' runner.

The checkout inventory describes this prepared revision. After intentional
source changes, rerun the relevant checks and regenerate the inventory before
claiming that the new files match this release. Historical archive-building
scripts describe their original file selection; the general packaging tools
can create a new archive from the current source tree.
