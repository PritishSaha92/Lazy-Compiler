# Phase 3 coverage additions

These development cases extend the existing regression suite with transport and
schedule combinations that were not directly isolated by its earlier fixtures.
They are synthetic cases derived from the published Emmy 0.3.11 semantics;
they are not organizer hidden tests. No production compiler code reads these
definitions, expected outputs, or expression descriptions.

## Required-vocabulary audit

The audit compared `TASK.md`, the installed typed node definitions, the backend,
and existing tests. Every unstarred statement already had a lowering and direct
or public-workload coverage: scalar statements, both loops and conditionals,
tiles, shared declarations, CTA/named/warp barriers, asynchronous copy groups,
shuffle merges, and shared-tree merges/broadcast. The additional cases isolate
important combinations within those operations rather than claiming new
required node kinds.

| Addition | Independent correctness condition |
|---|---|
| 4/8/16-byte asynchronous copies | Reverse each lane's contiguous chunk across three CTAs. `wait_group 1` exposes the first completed group while the second may remain in flight; `wait_group 0` drains the second. CTA barriers order peer consumption. |
| Swizzled shared writes | Inspect the physical shared layout with ordinary loads after B32/B64/B128/B128@7 stores. A host permutation of eight-element chunks supplies the reference; an inverse read cannot hide an incorrect write. |
| Dynamic shared allocation | Two differently aligned buffers total 49,284 bytes including padding. Distinct initialized cells at both ends of both arrays detect overlap and incorrect pool offsets. |
| Grouped unaligned loads | Four consecutive elements start at offset one in rows of width seven. The operation remains valid when a single vector instruction is not aligned. |
| Nonempty multiplication folds | Zero, one, and multiple iterations use the product identity, preserve a saved local state, and broadcast through an eight-lane shuffle or a 64-thread shared tree. FP32 powers of two allow exact host references from integer exponent sums. |

The eleven typed cases are defined in
[`test_transport_edges.py`](../tests/test_transport_edges.py) and
[`test_product_reduction.py`](../tests/test_product_reduction.py). Their concrete
inputs, expected arrays, IRs, launch metadata, and hash manifests were saved in
[`phase3/transport`](../artifacts/validation/phase3/transport/) and
[`phase3/products`](../artifacts/validation/phase3/products/) before their backend
evaluations. The initial transport check assembled both native and synchronous
variants: 18 assembly checks. The initial product check assembled and ran both
cases on the local GPU. The final regression reports determine which compiler
revision each release result supports.

The local pre-sm80 transport path explicitly replaces asynchronous copies with
synchronous loads/stores. It validates addressing and consumer barriers; native
copy completion is evaluated separately on the official runner. Neither a
numerical pass nor an assembly pass is a sanitizer certification.

The full local release attempt retains a failed gate: 183 assembly checks and
168 numerical executions succeeded, while `dynamic_pool_distinct_alignment`
was rejected before launch by a helper assumption that every pre-sm80 device
has only 48 KiB shared capacity. A read-only driver query found 64 KiB opt-in
capacity on this device, so the rejection identifies a harness defect. The
[local audit](../artifacts/validation/phase3/release-local-audit.json) preserves
the original failure and distinguishes the one unexecuted case from the 168
successful executions. The capability-query correction was promoted after five
CPU mock tests; it has not been validated through a local GPU launch. Further
kernel execution uses the official runner, as requested by the user.

## Unsigned input transport

The first full official run passed 209 of 213 cases. Four alignment fixtures
were rejected because the runner accepts signed integer argument descriptors,
but not unsigned descriptors. Merely reinterpreting `u64` input arrays as `i64`
was insufficient: the runner's numeric transport changed low bits of large
integers. That first follow-up passed the two `u32` cases, while the two `u64`
cases faulted; both failures remain recorded.

A separate bounded-address probe read four fixed `u64` values and returned
their four 16-bit words as FP32. Loaded input bits could not affect any address.
The old `i64` transport executed but returned two incorrect words, exactly
matching an intermediate float64 rounding model. Transporting the same bytes
as flat `i32` words passed all 16 output values. Every signed 32-bit word is
exactly representable in float64; input byte hashes, IR, expected arrays and
emitted PTX stayed identical. All four alignment fixtures subsequently passed
on the official runner with this word representation.

The [safe probe](../artifacts/validation/phase3/word-transport-diagnostic/manifest.json),
[transport manifest](../artifacts/validation/phase3/unsigned-word-transport/adaptation-manifest.json),
and [four-case official report](../artifacts/validation/phase3/unsigned-word-transport/remote.json)
retain this evidence. These are four reruns of existing logical cases, not four
new workloads. The original 209/213 report remains a failed gate; the combined
coverage joins its 209 successful cases to the four verified transport reruns.

[`check_release.py`](../tools/check_release.py) now performs this input-only
adaptation automatically. It retains original packages, records byte identity
and exact PTX identity, and still schedules 213 checks over 97 logical packages.
Unsigned outputs, by-value scalars and generated unsigned inputs are rejected
explicitly rather than silently reinterpreted. Seven
[CPU tests](../tests/test_unsigned_transport.py) cover descriptor preservation,
the full unsigned integer range, float64 round trips, failure behavior, actual
CLI equality and a mocked complete release workflow. The mocked workflow is
not numerical execution evidence.

```console
python tools/check_release.py --compiler compile.py --report-dir artifacts/validation/new-release
```

## Fresh scheduled workloads

[`expressions_phase3.py`](../tests/expressions_phase3.py) fixed six expressions and
their independent NumPy references before compiler evaluation. CPU PyTorch and
those references agreed on all 18 checks: normal inputs at seed offsets 37 and
89, and inputs scaled by eight at offset 137. Emmy generated five valid kernel
packages:

| Package | Schedule feature |
|---|---|
| `phase3_p01_three_axis_broadcast` | Independent broadcasts across three dimensions; exact FP32 pointwise output. |
| `phase3_f01_column_softmax` | Softmax over the non-innermost axis of a 97 by 13 input. |
| `phase3_f02_centered_column_l2` | Noncontiguous mean/reduction reuse for a 73 by 19 centered normalization. |
| `phase3_m01_batched_odd_matmul` | Batched `(3,17,33) @ (3,33,21)` contraction with ragged dimensions. |
| `phase3_a01_irregular_attention` | Forward attention with dimensions 17, 23, 43, and 11. |

The sixth expression, `torch.sigmoid(x).prod(-1)`, failed in Emmy's frontend:
`ValueError: cannot map fallback aten.prod as elementwise for output shape (37,)`.
The [captured failure](../artifacts/validation/phase3/generation-product-recheck.log)
remains visible. It is excluded from executable schedule counts and is not a
backend failure or a passing compiler test. The direct typed product fixtures
above test valid product IR without depending on that unsupported trace.

[`check_phase3_references.py`](../tools/check_phase3_references.py) checks traced
input names, ordering and shapes, recomputes packaged expected arrays, then seals
source and artifact hashes. The [schedule manifest](../artifacts/validation/phase3/schedule-manifest.json)
records five executable workloads and the one missing frontend package. No
tolerance or expression was changed after seeing backend results.

```console
python tools/check_phase3_references.py --verify
python -m unittest discover -s tests -p test_transport_edges.py -v
python -m unittest discover -s tests -p test_product_reduction.py -v
```

Set `EMMY_GPU_TESTS=1` for the local numerical methods. Runner tests of the five
scheduled packages should use `--references tests/expressions_phase3.py` with
the fixed normal seeds 37/89 and wide seed 137. They provide 15 numerical cases;
the 18 CPU reference checks are a different count and must not be added to GPU
coverage.
