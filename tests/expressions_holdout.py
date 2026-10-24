"""Untuned 23 October 2026 workload split, independent of contest hidden data.

These expressions are test-data inputs to Emmy, never inputs to our PTX backend.
The references use NumPy arithmetic and float64 reduction/dot intermediates;
pointwise exactness is required only where the FP32 expression is unambiguous.
Freeze this file and generated artifacts before comparing compiler policies.
"""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def dot(a, b):
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def centered_affine(x, weight, bias):
    x = x.astype(np.float64)
    centered = x - x.mean(axis=-1, keepdims=True)
    return (
        centered
        / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 0.002)
        * weight
        + bias
    ).astype(np.float32)


def normalized_residual(x, residual, weight):
    # Residual addition occurs before the reduction in the expression graph.
    combined = (x + residual).astype(np.float64)
    return (
        combined
        / np.sqrt((combined * combined).sum(axis=-1, keepdims=True) + 0.0003)
        * weight
    ).astype(np.float32)


OPEN = [
    dict(
        id="hold_p01_broadcast_tanh",
        category="pointwise",
        family="two-axis broadcast with nonlinear epilogue",
        code="x=torch.randn(23,47); w=torch.randn(47); b=torch.randn(23,1); torch.tanh(x*w+b)",
        ref=lambda x, w, b: [np.tanh(x * w + b)],
        rtol=1e-4,
        atol=1e-5,
    ),
    dict(
        id="hold_p02_absolute_difference_3d",
        category="pointwise",
        family="three-dimensional two-branch pointwise composition",
        code="x=torch.randn(3,19,37); y=torch.randn(3,19,37); torch.relu(x-y)+torch.relu(y-x)",
        ref=lambda x, y: [np.maximum(x - y, 0) + np.maximum(y - x, 0)],
        rtol=0,
        atol=0,
    ),
    dict(
        id="hold_p03_rational_gate",
        category="pointwise",
        family="odd-extent rational and sigmoid composition",
        code="x=torch.randn(2053); y=torch.randn(2053); torch.sigmoid(x)*(y/(1.0+x*x))",
        ref=lambda x, y: [(1 / (1 + np.exp(-x))) * (y / (np.float32(1) + x * x))],
        rtol=1e-4,
        atol=1e-5,
    ),
    dict(
        id="hold_r01_column_sum",
        category="reduction",
        family="noncontiguous column reduction",
        code="torch.randn(71,33).sum(0)",
        ref=lambda x: [x.astype(np.float64).sum(axis=0).astype(np.float32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=REDUCE_KNOBS,
    ),
    dict(
        id="hold_r02_row_mean_square",
        category="reduction",
        family="mean-square reduction over an odd row width",
        code="x=torch.randn(17,251); (x*x).mean(-1)",
        ref=lambda x: [(x.astype(np.float64) ** 2).mean(axis=-1).astype(np.float32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=REDUCE_KNOBS,
    ),
    dict(
        id="hold_r03_row_inner_product",
        category="reduction",
        family="odd-width two-input product reduction",
        code="x=torch.randn(13,257); y=torch.randn(13,257); (x*y).sum(-1)",
        ref=lambda x, y: [
            (x.astype(np.float64) * y.astype(np.float64))
            .sum(axis=-1)
            .astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-3,
        knobs=REDUCE_KNOBS,
    ),
    dict(
        id="hold_r04_column_max",
        category="reduction",
        family="odd-height noncontiguous maximum",
        code="torch.amax(torch.randn(129,11),dim=0)",
        ref=lambda x: [x.max(axis=0)],
        rtol=0,
        atol=0,
        knobs=REDUCE_KNOBS,
    ),
    dict(
        id="hold_m01_long_reduction",
        category="matmul",
        family="long-K narrow-output matrix product",
        code="torch.randn(24,1536) @ torch.randn(1536,40)",
        ref=lambda a, b: [dot(a, b).astype(np.float32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="hold_m02_tall_thin",
        category="matmul",
        family="tall-thin matrix product",
        code="torch.randn(192,40) @ torch.randn(40,12)",
        ref=lambda a, b: [dot(a, b).astype(np.float32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="hold_m03_wide_output",
        category="matmul",
        family="short-wide matrix product",
        code="torch.randn(12,48) @ torch.randn(48,224)",
        ref=lambda a, b: [dot(a, b).astype(np.float32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="hold_m04_bias_relu",
        category="matmul",
        family="matrix product with broadcast bias and activation",
        code="a=torch.randn(48,160); b=torch.randn(160,24); c=torch.randn(24); torch.relu(a@b+c)",
        ref=lambda a, b, c: [np.maximum(dot(a, b) + c, 0).astype(np.float32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="hold_f01_centered_affine",
        category="fused",
        family="mean-centered normalization with learned affine transform",
        code="x=torch.randn(11,197); w=torch.randn(197); b=torch.randn(197); (x-x.mean(-1,keepdim=True))/torch.sqrt(((x-x.mean(-1,keepdim=True))*(x-x.mean(-1,keepdim=True))).mean(-1,keepdim=True)+0.002)*w+b",
        ref=lambda x, w, b: [centered_affine(x, w, b)],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="hold_f02_residual_l2_weight",
        category="fused",
        family="residual addition before L2 normalization and broadcast weight",
        code="x=torch.randn(21,61); r=torch.randn(21,61); w=torch.randn(61); (x+r)/torch.sqrt(((x+r)*(x+r)).sum(-1,keepdim=True)+0.0003)*w",
        ref=lambda x, r, w: [normalized_residual(x, r, w)],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="hold_f03_weighted_softmax_sum",
        category="fused",
        family="scaled online normalizer feeding a weighted reduction",
        code="x=torch.randn(19,137); g=torch.randn(19,137); (torch.softmax(x*0.375,dim=-1)*g).sum(-1)",
        ref=lambda x, g: [
            (softmax(x.astype(np.float64) * 0.375) * g.astype(np.float64))
            .sum(axis=-1)
            .astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="hold_f04_rms_sigmoid_gate",
        category="fused",
        family="RMS normalization composed with sigmoid gating",
        code="x=torch.randn(7,311); torch.sigmoid(x)*x/torch.sqrt((x*x).mean(-1,keepdim=True)+0.002)",
        ref=lambda x: [
            (
                x.astype(np.float64)
                / (1 + np.exp(-x.astype(np.float64)))
                / np.sqrt(
                    (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + 0.002
                )
            ).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="hold_a01_long_context",
        category="attention",
        family="long-context narrow rectangular attention",
        code="q=torch.randn(8,24); kt=torch.randn(24,192); v=torch.randn(192,12); torch.softmax(q@kt/4.898979486,dim=-1)@v",
        ref=lambda q, kt, v: [
            (softmax(dot(q, kt) / 4.898979486) @ v.astype(np.float64)).astype(
                np.float32
            )
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="hold_a02_long_query",
        category="attention",
        family="long-query short-context rectangular attention",
        code="q=torch.randn(160,16); kt=torch.randn(16,24); v=torch.randn(24,20); torch.softmax(q@kt/4.0,dim=-1)@v",
        ref=lambda q, kt, v: [
            (softmax(dot(q, kt) / 4.0) @ v.astype(np.float64)).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="hold_a03_attention_residual",
        category="attention",
        family="rectangular attention with scaled residual epilogue",
        code="q=torch.randn(12,20); kt=torch.randn(20,80); v=torch.randn(80,28); r=torch.randn(12,28); (torch.softmax(q@kt*0.25,dim=-1)@v)+r*0.125",
        ref=lambda q, kt, v, r: [
            (
                softmax(dot(q, kt) * 0.25) @ v.astype(np.float64)
                + r.astype(np.float64) * 0.125
            ).astype(np.float32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
]
