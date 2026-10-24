# Emmy Kernel IR to PTX compiler

**Implementation:** [`compile.py`](compile.py) now provides a direct PTX backend.
Start with the [submission overview](SUBMISSION.md),
[feature/example/test index](docs/FEATURES.md), and [runnable examples](examples/README.md).
The optional extensions include FP16/BF16 tensor cores and an SM90 bulk-copy
path. Their exact supported forms and limits are explicit. The latest [release report](docs/PHASE6.md)
and [DeepInfer/MoK research review](docs/RESEARCH_REVIEW.md) describe the additional
ring-pipeline tests and architecture-admission correction.

Emmy (github.com/cloudrift-ai/emmy) compiles PyTorch to CUDA C++ and builds it with `nvcc`.
**Your task: write a compiler that takes Emmy's Kernel IR and emits PTX directly**, skipping the
CUDA C++ step. `ptxas` turns PTX into the same GPU binary ~19× faster than the full `nvcc`
pipeline — Emmy's autotuner compiles thousands of kernel variants, so that middle step matters.

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
| [`compile_file`](ptx_backend.py#L1685), [`compile`](ptx_backend.py#L1633) | Read the ABI and numerical contract, declare parameters/registers/shared memory, assemble the module. |
| [`statement`](ptx_backend.py#L1251), [`tile`](ptx_backend.py#L1449), [`loop`](ptx_backend.py#L1378) | Dispatch typed IR; preserve work ownership, bounds, lexical scope and accumulator lifetime. |
| [`expr`](ptx_backend.py#L739), [`operation`](ptx_backend.py#L788), [`convert`](ptx_backend.py#L439) | Lower typed arithmetic with explicit conversion, rounding and unsupported-operation diagnostics. |
| [`address`](ptx_backend.py#L1117), [`load`](ptx_backend.py#L1156), [`store`](ptx_backend.py#L1177) | Flatten indices and emit global/shared memory operations; vectorize only with sufficient alignment evidence. |
| [`shuffle`](ptx_backend.py#L1543), [`tree`](ptx_backend.py#L1580), [`sync`](ptx_backend.py#L1519) | Preserve subgroup membership, ordered coupled-state merges, shared-memory barriers and result broadcast. |
| [`TensorLowering`](ptx_tensor.py) | Lower explicit m16n8k16 fragments, operand transport, MMA, guarded stores and epilogues. |
| [`BulkLowering`](ptx_bulk.py) | Prove aligned box-copy bounds, emit native bulk transactions and preserve mbarrier phases. |

The prescribed launch and schedule are retained. Optimizations are bounded:
schedule-hinted loop unrolling and, only under a nonzero tolerance, compensation
for eligible independent FP32 dots. Exact or missing tolerances retain ordinary
FP32 dot arithmetic. Operation/type support is explicit; see
[the type rules](docs/TYPE_SUPPORT.md), [lowering contracts](docs/TRANSFORMATION_CONTRACTS.md),
[validation evidence](docs/VALIDATION.md) and the
[read-aloud video script](docs/VIDEO_SCRIPT.md). Passing the public set is a
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

Submit at [riftstack.ai/hackathon/submit](https://riftstack.ai/hackathon/submit) before
**23:59 AoE on Tuesday, October 27**. A submission is:

- a **public GitHub repository** with `compile.py` at its root, runnable as
  `python compile.py path/to/kernel.ir.json > out.ptx`;
- a **video of 3 to 5 minutes** in which you walk through your compiler in your own voice
  (YouTube, Vimeo or Google Drive, shared so that anyone with the link can watch it).

Teams of up to three submit once and list every member. You can submit again until the
deadline; we review your latest entry. AI coding assistants are allowed, but your video has to
show that you understand how your compiler works.

## How compilers are evaluated

We run your `compile.py` on every kernel of the **hidden set** — the same five categories as the
open problems, with different shapes and ops — then assemble the PTX with `ptxas`, run it on the
runner's GPU under each problem's `meta.json` launch, and compare the outputs with the expected
ones within the problem's tolerance. This is the pipeline `submit.py --compiler` runs, so passing
the open set that way is the best preparation. Keep compiling to seconds per kernel: a time
limit may apply.

1. **Correctness on the hidden set, by category.** Passing categories 1 and 2 is the bar
   ([MVP.md](MVP.md)); each further category is what sets submissions apart.
2. **Code review** of your compiler is the first tiebreak. The IR → PTX lowering has to be your
   own — see the rule in [TASK.md](TASK.md).
3. **Speed** of the generated kernels is the last tiebreak.

The video decides prize eligibility, as described above.

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
