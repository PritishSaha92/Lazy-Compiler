These are the unchanged audit scripts retained with the two oracle reports.
Their hashes appear in the reports. They were run in the original workspace
under `research/competitors-2026-10-28/own-probes`, so their root discovery
reflects that layout. They are evidence source, not submission runtime.

The first script audits the original I64 FP64 fixtures and six loop fixtures;
its mathematical result did not establish lossless I64 runner transport.
The supplemental script audits the replacement two-I32-word FP64 fixtures
and explicitly reproduces the runner numeric conversion. The original report
and source are preserved, including the earlier generator hash.
