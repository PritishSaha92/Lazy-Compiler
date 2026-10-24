# Preparing the required 3-5 minute walkthrough

Record this in your own voice, using explanations you can defend. The contest
permits AI assistance but requires your understanding. This outline is rehearsal
material, not a script to read without understanding. Be candid about the
assistance used and which design decisions and tests you examined yourself.

Use the updated [609-word script](VIDEO_SCRIPT.md), [feature index](FEATURES.md),
[phase-6 results](PHASE6.md) and [runtime release pointer](../artifacts/release/phase6/latest.json).
The current walkthrough includes tensor-core fragment transport and the combined
bulk-copy/MMA pipeline. Record in your own voice and inspect the cited functions.

The following core-only demonstration outline is retained from phase 4; the
current script is the complete 3-5 minute route.

## Suggested demonstration

**0:00-0:35 — The contract.** From the repository directory, show
`python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx`.
Explain that the input argument is the path to an existing Kernel IR JSON file.
In Windows PowerShell 5.1, use `cmd /c "python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx"` so redirection preserves ASCII bytes.
Explain that Emmy supplies scheduled typed IR, this program emits PTX directly,
and `ptxas` performs physical register allocation and final machine-code assembly.
Show that the compiler reads the IR and metadata, not `.ref.cu` or `expected.npz`.

**0:35-1:15 — One thread's work.** Open the smallest pointwise IR. Trace the Tile
cell index, the bounds guard, the two input addresses and the vector output.
Explain why 256 launched threads do not imply 256 valid work cells. Show the
pointer parameter order from `meta.json` and one generated load/store sequence.

**1:15-2:15 — Reductions and softmax.** Trace accumulator initialization before a
strided loop, the per-thread partial, the shuffle butterfly, the shared tree and
the final broadcast. Then show `Accum.base` in online softmax: the maximum and
normalizer form a coupled state, so the old normalizer must be rescaled before
adding a new contribution. Explain why the merge program's statement order and
the final broadcast both matter.

**2:15-3:05 — Matmul and state lifetime.** Show `Init` outside the slab loop and
`seed=False` inside it. Explain that reseeding the inner dot would discard prior
slabs. Show native `cp.async`, commit, wait and the separate consumer barrier.
Point out the dynamic shared allocation and the split-K atomic output reset.

**3:05-4:10 — Evidence and one discovered bug.** Demonstrate the test command and
open the [213-check regression report](../artifacts/validation/phase4/release-remote/summary.json),
[29-check semantic report](../artifacts/validation/phase4/semantic-final/summary.json)
and [separate 90-check corpus report](../artifacts/validation/phase4/selector-correctness-retry01/summary.json).
Explain that these are 332 checks across 150 packages, not 332 distinct programs.
The earlier incomplete corpus attempt is retained and excluded from this count.
All phase 4 GPU checks used the official runner; compilation runs on the CPU.
Distinguish assembly, output correctness and indicative timing. Explain a concrete regression: unsigned 32-bit
`1431655766 * 6` wraps to four; division by three gives an odd element offset.
A pre-overflow divisibility fact cannot justify a vector load at that address.
Show one independent interaction fixture: a ragged row with changing alignment,
or a seven-stage asynchronous pipeline whose two slots must be drained correctly.
Mention the operation/type admission table and a diagnostic for an unsupported
pair; recognizing a buffer dtype does not make every arithmetic operation legal.

**4:10-4:45 — Numeric policy and limits.** Explain the cancellation stress case
and the measured strict/FMA/FP64/compensated alternatives. State the selected
policy and its cost using the source-labelled performance report. The selector
pilot changes only existing unroll budgets and found too few distinct PTX
choices to justify training or comparative timing. No learned policy is in the
submitted compiler; the proposed offline replay protocol is not trained
coding-model RL. Explain why this negative result keeps the baseline. End with supported categories
and actual evidence; do not claim hidden results or a guaranteed placement.

## Questions to answer without reading code

- How is a buffer pointer different from a scalar loaded through it?
- Why does the parameter order come from metadata?
- What survives a loop iteration, and what resets before a nested reduction?
- Why must all components be shuffled before merging a coupled state?
- Which lanes belong in a shuffle mask after threads take different loop counts?
- Why can a copy wait require a separate block barrier?
- Why is a zero-filled output insufficient to detect missing stores?
- What numerical behavior can FMA or a wider accumulator change?
- Why can delaying an FP16/BF16 rounding step change a nested expression?
- How do you check that a transport workaround preserves the kernel's input bytes?
- Why does passing public inputs not prove hidden-set correctness?

Publish your own public repository and shareable video when ready. The
organizer's development runner is not the formal competition submission form.
