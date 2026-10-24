"""Fresh schedule probes fixed before any PTX or GPU feedback, 23 October.

Used only by test-data generation. The submitted backend never reads these
expressions or mathematical references. Inputs retain the generator's seeded
FP32 semantics; reduction and matrix references use independent NumPy FP64.
"""

import numpy as np


def softmax(values, axis):
    values = np.asarray(values, dtype=np.float64)
    exponential = np.exp(values - values.max(axis=axis, keepdims=True))
    return exponential / exponential.sum(axis=axis, keepdims=True)


OPEN = [
    dict(
        id="phase3_p01_three_axis_broadcast",
        category="pointwise",
        family="three-dimensional independent axis broadcasts",
        code="x=torch.randn(7,1,13); y=torch.randn(1,9,1); z=torch.randn(7,9,13); torch.relu(x+y-z)",
        ref=lambda x, y, z: [np.maximum(x + y - z, 0)],
        rtol=0,
        atol=0,
    ),
    dict(
        id="phase3_r01_product_sigmoid",
        category="reduction",
        family="nonempty multiplicative reduction with nontrivial identity",
        code="x=torch.randn(37,11); torch.sigmoid(x).prod(-1)",
        ref=lambda x: [
            (1 / (1 + np.exp(-x.astype(np.float64)))).prod(axis=-1).astype(np.float32)
        ],
        rtol=1e-4,
        atol=1e-7,
        knobs="WORK=t128,REDUCE=coop",
    ),
    dict(
        id="phase3_f01_column_softmax",
        category="fused",
        family="non-innermost online normalizer with ragged column count",
        code="x=torch.randn(97,13); torch.softmax(x,dim=0)",
        ref=lambda x: [softmax(x, axis=0).astype(np.float32)],
        rtol=1e-3,
        atol=1e-5,
    ),
    dict(
        id="phase3_f02_centered_column_l2",
        category="fused",
        family="noncontiguous centered normalization with reduction reuse",
        code="x=torch.randn(73,19); (x-x.mean(0,keepdim=True))/torch.sqrt(((x-x.mean(0,keepdim=True))*(x-x.mean(0,keepdim=True))).sum(0,keepdim=True)+0.01)",
        ref=lambda x: [
            (
                (x.astype(np.float64) - x.mean(axis=0, keepdims=True, dtype=np.float64))
                / np.sqrt(
                    (
                        (
                            x.astype(np.float64)
                            - x.mean(axis=0, keepdims=True, dtype=np.float64)
                        )
                        ** 2
                    ).sum(axis=0, keepdims=True)
                    + 0.01
                )
            ).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="phase3_m01_batched_odd_matmul",
        category="matmul",
        family="three-dimensional batched contraction with ragged dimensions",
        code="a=torch.randn(3,17,33); b=torch.randn(3,33,21); a@b",
        ref=lambda a, b: [
            (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="phase3_a01_irregular_attention",
        category="attention",
        family="online attention with all four matrix dimensions irregular",
        code="q=torch.randn(17,23); kt=torch.randn(23,43); v=torch.randn(43,11); torch.softmax(q@kt*0.125,dim=-1)@v",
        ref=lambda q, kt, v: [
            (
                softmax(q.astype(np.float64) @ kt.astype(np.float64) * 0.125, axis=-1)
                @ v.astype(np.float64)
            ).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
]
