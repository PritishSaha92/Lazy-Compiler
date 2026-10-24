"""Fresh schedule/shape cases for tools/generate.py; never imported by the compiler.

The original generator may call Emmy's CUDA renderer to extract matching launch
metadata. That is test-data generation only; compile.py still lowers typed IR
directly and cannot access these expressions or the generated references.
"""

import numpy as np


def softmax(x):
    x = x.astype(np.float64)
    values = np.exp(x - x.max(-1, keepdims=True))
    return values / values.sum(-1, keepdims=True)


OPEN = [
    dict(
        id="stress_pointwise_prime",
        code="x=torch.randn(257); y=torch.randn(257); (x-y)*y",
        ref=lambda x, y: [(x - y) * y],
        rtol=0,
        atol=0,
    ),
    dict(
        id="stress_pointwise_2d",
        code="x=torch.randn(17,65); torch.tanh(x) + torch.sigmoid(x)",
        ref=lambda x: [np.tanh(x) + 1 / (1 + np.exp(-x))],
        rtol=1e-4,
        atol=1e-5,
    ),
    dict(
        id="stress_sum_prime",
        code="torch.randn(1003).sum()",
        ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)],
        rtol=1e-4,
        atol=1e-4,
        knobs="WORK=t128,REDUCE=coop",
    ),
    dict(
        id="stress_row_sum_tail",
        code="torch.randn(7,513).sum(-1)",
        ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)],
        rtol=1e-4,
        atol=1e-4,
        knobs="WORK=t128,REDUCE=coop",
    ),
    dict(
        id="stress_max_tail",
        code="torch.amax(torch.randn(9,73),dim=-1)",
        ref=lambda x: [x.max(-1)],
        rtol=0,
        atol=0,
        knobs="WORK=t128,REDUCE=coop",
    ),
    dict(
        id="stress_softmax_tail",
        code="torch.softmax(torch.randn(7,93),dim=-1)",
        ref=lambda x: [softmax(x).astype(np.float32)],
        rtol=1e-3,
        atol=1e-5,
    ),
    dict(
        id="stress_rmsnorm_tail",
        code="x=torch.randn(9,73); x/torch.sqrt((x*x).mean(-1,keepdim=True)+0.0001)",
        ref=lambda x: [
            (
                x
                / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 0.0001)
            ).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="stress_matmul_rect",
        code="torch.randn(64,96) @ torch.randn(96,80)",
        ref=lambda a, b: [
            (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="stress_matmul_tail",
        code="torch.randn(33,17) @ torch.randn(17,29)",
        ref=lambda a, b: [
            (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="stress_attention_rect",
        code="q=torch.randn(16,16); kt=torch.randn(16,32); v=torch.randn(32,8); torch.softmax(q@kt/4.0,dim=-1)@v",
        ref=lambda q, kt, v: [
            (
                softmax(q.astype(np.float64) @ kt.astype(np.float64) / 4.0)
                @ v.astype(np.float64)
            ).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
]
