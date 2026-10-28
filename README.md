# Lazy Compiler

**Implementation:** [`compile.py`](compile.py) now provides a direct PTX backend.
Start with the [submission overview](SUBMISSION.md),
[feature/example/test index](docs/FEATURES.md), and [runnable examples](examples/README.md).
Optional extensions include FP16/BF16 tensor cores, FP32 fragment algebra and
row reductions, and an SM90 bulk-copy path. The current
[phase-9 revision](docs/PHASE9.md) also adds direct GELU and tanh-GELU operations.
Supported forms, numerical boundaries and diagnostics are explicit.

**Phase-9 validation passed all 15 public problems and 67 additional feature
packages** on the organizers' GPU. CPU/assembler discovery passed 129 methods,
with 19 explicit local-GPU skips. The public result includes one successful retry
after a connection reset. Separately, the final generated-composition study
passed **21/24 mathematical checks**: three [FP16 stage-rounding conflicts](docs/PHASE9.md#generated-compositions-and-retained-numerical-conflict)
remain failures, with the same outcome on the baseline compiler. See the
[source-bound release summary](artifacts/validation/phase9/release-summary.json)
and [phase-9 report](docs/PHASE9.md) for scope and retained failed attempts.

This Python backend translates Emmy's scheduled Kernel IR directly into PTX.
Emmy 0.3.11 supplies parsing and typed IR structures; this compiler implements
the lowering. NVIDIA's `ptxas` assembles the emitted PTX downstream.

You do **not** need a GPU. A hosted runner at `http://runner.riftstack.ai:48701` (the default in `submit.py`) compiles
your PTX with `ptxas`, executes it on real hardware, and tells you whether the output is correct. See [docs/API.md](docs/API.md).

## Compiler architecture

The runtime is [`compile.py`](compile.py), [`ptx_backend.py`](ptx_backend.py),
[`ptx_tensor.py`](ptx_tensor.py), [`ptx_bulk.py`](ptx_bulk.py),
and [`requirements.txt`](requirements.txt). The review bundle also includes
examples, tests, tools and documentation. Compilation needs Python and pinned
Emmy 0.3.11; it does not need Torch, CUDA or a GPU. Torch is used only by the
separate development generator.

The pipeline is **Kernel IR JSON + neighboring `meta.json` -> Emmy typed objects
-> this backend's statement/expression lowering -> PTX 8.0**. Emmy supplies
parsing and IR definitions. This backend implements instruction selection,
addressing, control flow and cooperation directly; it does not call Emmy's CUDA
renderer or another compiler backend. The entry point writes PTX only after the
whole module succeeds, with diagnostics on stderr.

| Read in this order | Responsibility |
|---|---|
| [`compile_file`](ptx_backend.py#L1821), [`compile`](ptx_backend.py#L1758) | Read the ABI and numerical contract, declare parameters/registers/shared memory, assemble the module. |
| [`statement`](ptx_backend.py#L1375), [`tile`](ptx_backend.py#L1574), [`loop`](ptx_backend.py#L1496) | Dispatch typed IR; preserve work ownership, bounds, lexical scope and accumulator lifetime. |
| [`expr`](ptx_backend.py#L803), [`operation`](ptx_backend.py#L894), [`convert`](ptx_backend.py#L470) | Lower typed arithmetic with explicit conversion, rounding and unsupported-operation diagnostics. |
| [`address`](ptx_backend.py#L1241), [`load`](ptx_backend.py#L1280), [`store`](ptx_backend.py#L1301) | Flatten indices and emit global/shared memory operations; vectorize only with sufficient alignment evidence. |
| [`shuffle`](ptx_backend.py#L1668), [`tree`](ptx_backend.py#L1705), [`sync`](ptx_backend.py#L1644) | Preserve subgroup membership, ordered coupled-state merges, shared-memory barriers and result broadcast. |
| [`TensorLowering`](ptx_tensor.py) | Lower explicit m16n8k16 fragments, MMA, guarded stores, elementwise fragment algebra and ordered row reductions. |
| [`BulkLowering`](ptx_bulk.py) | Prove aligned box-copy bounds, emit native bulk transactions and preserve mbarrier phases. |

GPU builtins retain their signed/unsigned types. Unstamped mixed-integer
arithmetic stays in the integer family, and expression shifts take their type
from the promoted left operand. Auxiliary-thread examples use explicit role
guards and barriers. Loops use signed 32-bit induction bindings, capture explicit
ends once, and evaluate live increments after the body. Named floating remainder
uses power-of-two reduction instead of a rounded quotient. See the
[feature contracts](docs/FEATURES.md) for boundaries and examples.

The prescribed launch and schedule are retained. `.maxntid` exposes the supplied
block size to the assembler; the [paired experiment](docs/PHASE8.md#performance-experiment)
is historical evidence against our previous compiler. No new phase-9 speedup is claimed. Other optimizations are bounded:
schedule-hinted loop unrolling and, only under a nonzero tolerance, compensation
for eligible independent FP32 dots. Exact or missing tolerances retain ordinary
FP32 dot arithmetic. Operation/type support is explicit; see
[the type rules](docs/TYPE_SUPPORT.md), [lowering contracts](docs/TRANSFORMATION_CONTRACTS.md),
[validation evidence](docs/VALIDATION.md). Passing the public set is a
baseline check; additional generated workloads test shapes and operation mixes
outside it.

## Requirements

- **Python 3.12 or newer.** The Kernel IR only deserializes through Emmy's own classes:

  ```bash
  pip install emmy-ml==0.3.11       # no torch, no CUDA, no GPU needed
  ```

  This alone reads every problem IR. Torch is **not** needed — it is only required to
  *generate* new IRs from PyTorch expressions, which is optional.

- Your submission is a **program**, not PTX files. It must run as:

  ```bash
  python compile.py path/to/kernel.ir.json > out.ptx
  ```

  This contract matters: submissions are evaluated against a **hidden set** of Kernel IRs in the
  same categories as the open problems. Hardcoding the open set scores nothing.

## Five-minute start

Run these commands from this repository's root:

```console
python -m pip install -r requirements.txt
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json
python submit.py p00_scalar_add --compiler compile.py
python -m unittest discover -s tests -p test_ring_pipeline.py -v
```

Compilation prints PTX to stdout; `submit.py` runs it on the organizers' GPU.
The test command uses CPU references and the assembler when available, without
launching your GPU. For output-file redirection, including the PowerShell 5.1
encoding caveat, see [the example commands](examples/README.md).

Then open `problems/p00_scalar_add/` and look at three files:

| file | what it is |
|---|---|
| `*.ir.json` | the Kernel IR — **the input to your compiler** |
| `*.ir.txt` | the same IR, pretty-printed for reading |
| `*.ref.cu` | Emmy's generated CUDA C++ — the reference answer, in a different target language |

Load an IR in Python:

```python
import json
from emmy.compiler.graph import Graph
g = Graph.from_dict(json.load(open("problems/r01_array_sum/k_sum_1_reduce.ir.json")))
body = g.nodes["sum_1"].op.body     # a tree of typed statements: Tile, StridedLoop, Load, ...
```

Your compiler walks that tree and prints PTX. Start with the common node kinds in the short table —
see [TASK.md](TASK.md) for the categories and the node-by-node translation table.

## The problems

15 open problems in 5 categories, easy → hard: **pointwise → reductions → matmul → fused →
attention-like**. Each is a folder under `problems/` with the IR, the reference, a launch
descriptor (`meta.json` — you never guess grid/block), and expected outputs (`expected.npz`).

**MVP: your compiler passes categories 1 and 2.** See [MVP.md](MVP.md).

Problems are scheduled for `sm_89`, and the runner is an **RTX 5090 (`sm_120`)**, where PTX written for `sm_89`
runs unchanged; [problems/PROVENANCE.md](problems/PROVENANCE.md)
says how they were generated and checked. The hidden evaluation set contains more problems of the *same kinds* —
different shapes and ops, no surprise concepts.

## Submitting

Submit at [riftstack.ai/hackathon/submit](https://riftstack.ai/hackathon/submit).
Check the organizer's current deadline and submission instructions there. The
repository and project video are the entry; a local ZIP is not required to push
this checkout. A submission includes:

- a **public GitHub repository** with `compile.py` at its root, runnable as
  `python compile.py path/to/kernel.ir.json > out.ptx`;
- a **video of 3 to 5 minutes** in which you walk through your compiler in your own voice
  (YouTube, Vimeo or Google Drive, shared so that anyone with the link can watch it).

Teams of up to three submit once and list every member. You can submit again until the
deadline; we review your latest entry. AI coding assistants are allowed, but your video has to
show that you understand how your compiler works.

## How compilers are evaluated

A later organizer clarification, supplied by the entrant, says there is no
strict winning order: review considers **versatility, performance, design
quality, novelty and clarity**. The public problem set is a baseline, and
submissions are reviewed manually. The original starter's ordered
correctness/review/speed tiebreak description is retained only as historical
guidance, not presented as the current scoring formula.

Hidden kernels use the same categories with different shapes and operations.
The compiler must emit PTX for the supplied IR and launch metadata, and the
runner assembles and checks outputs against the problem's tolerance. Passing
the public examples does not establish hidden-set correctness or a ranking.

The IR-to-PTX translation must be the submission's own implementation; see
[TASK.md](TASK.md). The project video should demonstrate the entrant's
understanding of that implementation. Check the organizer's current submission
instructions for eligibility and deadline details.

## Repo map

```
problems/     the open set (15 problems)
examples/     hand-written PTX with comments mapping each block back to the IR
submit.py     one-command test: compile + run + check against expected outputs
docs/API.md   the runner API, if you want to call it directly
tools/        how the problems were generated (emmy compile -c ... --ir cuda)
TASK.md       the task, the categories, the CLI contract
MVP.md        the minimum bar
```

## Development

Development and code review were AI-assisted. See [the checkout notes](docs/CHECKOUT.md)
for source provenance and the checks on this repository.
