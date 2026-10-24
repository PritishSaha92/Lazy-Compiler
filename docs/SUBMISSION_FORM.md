# Submission form text

These answers describe the current phase-6 release. The 44 runner checks are
the 15 public problems plus 29 additional feature cases, not hidden-set results.

## Tech stack

Python 3.12+, Emmy 0.3.11 for parsing and typed IR structures, and NumPy for
reference calculations. I used Python's unittest framework, NVIDIA's ptxas for
assembly checks, and the organizers' GPU runner for correctness testing.
PyTorch was used during test generation; the compiler itself does not need
PyTorch or a GPU to run. AI tools assisted with development and code review.

## Approach

I built a direct backend from Emmy's scheduled Kernel IR to PTX. Emmy handles
deserialization, while my code translates expressions, memory accesses, loops,
reductions and synchronization into PTX instructions. The lowering follows the
IR's operations and types, so different shapes and operation combinations use
the same code paths.

My main priority was preserving the supplied schedule and numerical behavior.
That meant checking accumulator lifetimes, coupled reduction states, datatype
conversions and asynchronous-copy synchronization. The core lowering is separate
from two optional modules: one for FP16/BF16 tensor cores and another for aligned
SM90 bulk copies with transaction barriers. Unsupported combinations receive
explicit diagnostics.

Testing drove the implementation. The final release passed all 15 public
problems and 29 additional feature cases on the organizers' runner, including
ragged shapes and repeated ring-buffer reuse across copy-and-compute stages.
I also included independent references, regression tests, runnable examples and
an architecture guide. A detailed review of DeepInfer and Mixture of Kittens
informed stronger lifetime and compilation-isolation tests. Supported features
and known numerical limitations are documented alongside the results.
