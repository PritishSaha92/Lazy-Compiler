# Source checkout

This repository contains the phase-7 compiler, examples, tests and technical
documentation. The five runtime files are byte-identical to the release that
passed 15 public problems and 41 additional checks on the organizers' GPU.
See [PHASE7.md](PHASE7.md) for the results and their limits.

The source checkout omits presentation drafts and the scratch `out.ptx` file.
Python caches and local compiler outputs are ignored. The Git history is
preserved. Development and code review were AI-assisted.

`.gitattributes` disables automatic newline conversion so Git preserves the
source and evidence hashes on both Windows and POSIX checkouts.

`REVIEW_MANIFEST.json` records the files in this checkout, excluding Git
metadata, caches, local scratch files and the checkout verification reports.
It is a checkout inventory, not the inventory of the earlier ZIP. The original
[archive inventory](../artifacts/validation/phase7/source-archive-inventory.json)
and [extraction report](../artifacts/validation/phase7/review-bundle-verification.json)
remain available as provenance. Historical validation reports retain their
original paths, hashes and outcomes.

Run these commands from the repository root:

```console
python tools/check_checkout.py --report artifacts/validation/phase7/checkout/verification.json
python tools/run_regressions.py --report artifacts/validation/phase7/checkout/cpu-regressions.json
```

The checkout check verifies source hashes, documentation links and the actual
CLI output for 20 retained examples. The regression suite uses `ptxas` for
assembly checks when installed; local-GPU tests remain opt-in. These commands
do not contact the organizers' runner.

The checkout inventory describes this prepared revision. After intentional
source changes, rerun the relevant checks and regenerate the inventory before
claiming that the new files match this release. Historical archive-building
scripts describe their original file selection; the general packaging tools
can create a new archive from the current source tree.
