# Source checkout

This repository contains the current phase-8 compiler, examples, tests and
technical documentation. Use this checkout for the public repository; a new
ZIP is not required. See [PHASE8.md](PHASE8.md) for the changes and the
[current release summary](../artifacts/validation/phase8/release-summary.json)
for source-bound validation status.

The source checkout omits presentation drafts and the scratch `out.ptx` file.
Python caches and local compiler outputs are ignored. Development and code
review were AI-assisted. Git metadata is not part of the checkout inventory.

`.gitattributes` disables automatic newline conversion so Git preserves the
source and evidence hashes on both Windows and POSIX checkouts.

`REVIEW_MANIFEST.json` uses the version-2 checkout schema. It records the current
file inventory and runtime hashes, excluding Git metadata, caches, local scratch
files and checkout verification reports. Its `current_validation` references
bind the current runtime to its release report and 20 retained CLI expectations.
It is not the inventory of the earlier ZIP. The original
[archive inventory](../artifacts/validation/phase7/source-archive-inventory.json)
and [extraction report](../artifacts/validation/phase7/review-bundle-verification.json)
remain available as provenance. Their runtime hashes describe phase 7, not the
changed phase-8 backend. Historical validation reports retain their original
paths, hashes and outcomes; the verifier checks that provenance separately.

Run these commands from the repository root:

```console
python tools/check_checkout.py
python tools/run_regressions.py --report artifacts/validation/checkout-current/cpu-regressions.json
```

The checkout check writes to
`artifacts/validation/checkout-current/verification.json` by default. It verifies
source hashes, documentation links and actual CLI output for 20 retained
examples. Uniform LF output may be compared with retained Windows CRLF after
that explicit conversion; all other bytes must match. Use a new `--report` path
when repeating a check, because previous reports are not overwritten.
The regression suite uses `ptxas` for
assembly checks when installed; local-GPU tests remain opt-in. These commands
do not contact the organizers' runner.

The checkout inventory describes this prepared revision. After intentional
source changes, rerun the relevant checks and regenerate the inventory before
claiming that the new files match this release. Historical archive-building
scripts describe their original file selection; the general packaging tools
can create a new archive from the current source tree.
