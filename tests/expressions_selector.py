"""Predeclared 30-workload unroll-selector pilot; generation/reference use only.

Each expression/schedule family has two shapes and belongs to exactly one split.
Seeds are input variants within a workload, never train/test split keys.
"""
import numpy as np


def f64(x):
    return np.asarray(x, dtype=np.float64)


def softmax(x):
    x = f64(x)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def dot(a, b):
    return f64(a) @ f64(b)


def centered_column(x):
    centered = f64(x) - f64(x).mean(axis=0, keepdims=True)
    return centered / np.sqrt((centered * centered).sum(axis=0, keepdims=True) + 0.01)


OPEN = []


def family(name, split, category, sizes, code, reference, *, knobs="", exact=False, atol=None):
    for index, shape in enumerate(sizes):
        OPEN.append(dict(
            id=f"selector_{name}_{index}", family=name, split=split,
            category=category, schedule_family=f"{name}:{knobs or 'emmy-default'}",
            code=code(*shape), ref=reference, knobs=knobs,
            rtol=0 if exact else 1e-3, atol=(atol if atol is not None else (0 if exact else (1e-3 if category == "matmul" else 1e-4))),
        ))


family("affine_relu", "train", "pointwise", [(31, 257), (127, 1025)],
       lambda m, n: f"x=torch.randn({m},{n}); w=torch.randn({n}); b=torch.randn({m},1); torch.relu(x*w+b)",
       lambda x, w, b: [np.maximum(x * w + b, 0)], exact=True)
family("row_dot", "train", "reduction", [(17, 127), (63, 1025)],
       lambda m, n: f"x=torch.randn({m},{n}); y=torch.randn({m},{n}); (x*y).sum(-1)",
       lambda x, y: [(f64(x) * f64(y)).sum(axis=-1).astype(np.float32)], knobs="WORK=t128,REDUCE=coop")
family("plain_gemm", "train", "matmul", [(48, 96, 64), (192, 256, 128)],
       lambda m, k, n: f"a=torch.randn({m},{k}); b=torch.randn({k},{n}); a@b",
       lambda a, b: [dot(a, b).astype(np.float32)])
family("weighted_rms", "train", "fused", [(17, 193), (64, 769)],
       lambda m, n: f"x=torch.randn({m},{n}); w=torch.randn({n}); x/torch.sqrt((x*x).mean(-1,keepdim=True)+0.002)*w",
       lambda x, w: [(f64(x) / np.sqrt((f64(x)**2).mean(axis=-1, keepdims=True) + 0.002) * w).astype(np.float32)])
family("scaled_attention", "train", "attention", [(12, 24, 64, 16), (48, 32, 160, 32)],
       lambda q, d, s, v: f"q=torch.randn({q},{d}); kt=torch.randn({d},{s}); v=torch.randn({s},{v}); torch.softmax(q@kt*0.125,dim=-1)@v",
       lambda q, kt, v: [(softmax(dot(q, kt) * 0.125) @ f64(v)).astype(np.float32)])

family("rational_gate", "validation", "pointwise", [(4099,), (32771,)],
       lambda n: f"x=torch.randn({n}); y=torch.randn({n}); torch.sigmoid(x)*(y/(1.0+x*x))",
       lambda x, y: [(1 / (1 + np.exp(-f64(x))) * (f64(y) / (1 + f64(x)**2))).astype(np.float32)])
family("column_max", "validation", "reduction", [(97, 13), (513, 65)],
       lambda m, n: f"torch.amax(torch.randn({m},{n}),dim=0)",
       lambda x: [x.max(axis=0)], knobs="WORK=t128,REDUCE=coop", exact=True)
family("bias_relu_gemm", "validation", "matmul", [(33, 65, 47), (96, 192, 160)],
       lambda m, k, n: f"a=torch.randn({m},{k}); b=torch.randn({k},{n}); c=torch.randn({n}); torch.relu(a@b+c)",
       lambda a, b, c: [np.maximum(dot(a, b) + c, 0).astype(np.float32)])
family("weighted_softmax", "validation", "fused", [(19, 137), (63, 521)],
       lambda m, n: f"x=torch.randn({m},{n}); g=torch.randn({m},{n}); (torch.softmax(x*0.375,dim=-1)*g).sum(-1)",
       lambda x, g: [(softmax(f64(x) * 0.375) * f64(g)).sum(axis=-1).astype(np.float32)])
family("residual_attention", "validation", "attention", [(17, 20, 48, 12), (64, 40, 128, 24)],
       lambda q, d, s, v: f"q=torch.randn({q},{d}); kt=torch.randn({d},{s}); v=torch.randn({s},{v}); r=torch.randn({q},{v}); torch.softmax(q@kt*0.125,dim=-1)@v+r*0.125",
       lambda q, kt, v, r: [(softmax(dot(q, kt) * 0.125) @ f64(v) + f64(r) * 0.125).astype(np.float32)])

family("absolute_difference", "final", "pointwise", [(3, 19, 37), (7, 31, 129)],
       lambda b, m, n: f"x=torch.randn({b},{m},{n}); y=torch.randn({b},{m},{n}); torch.relu(x-y)+torch.relu(y-x)",
       lambda x, y: [np.maximum(x - y, 0) + np.maximum(y - x, 0)], exact=True)
family("sigmoid_product", "final", "reduction", [(23, 11), (65, 23)],
       lambda m, n: f"x=torch.randn({m},{n}); torch.exp(torch.log(torch.sigmoid(x*0.125)).sum(-1))",
       lambda x: [(1 / (1 + np.exp(-f64(x) * 0.125))).prod(axis=-1).astype(np.float32)],
       knobs="WORK=t128,REDUCE=coop", atol=1e-12)
family("batched_gemm", "final", "matmul", [(2, 17, 33, 21), (3, 48, 96, 64)],
       lambda b, m, k, n: f"a=torch.randn({b},{m},{k}); b=torch.randn({b},{k},{n}); a@b",
       lambda a, b: [dot(a, b).astype(np.float32)])
family("centered_column_l2", "final", "fused", [(73, 19), (257, 67)],
       lambda m, n: f"x=torch.randn({m},{n}); (x-x.mean(0,keepdim=True))/torch.sqrt(((x-x.mean(0,keepdim=True))*(x-x.mean(0,keepdim=True))).sum(0,keepdim=True)+0.01)",
       lambda x: [centered_column(x).astype(np.float32)])
family("score_bias_attention", "final", "attention", [(13, 23, 59, 17), (48, 40, 129, 32)],
       lambda q, d, s, v: f"q=torch.randn({q},{d}); kt=torch.randn({d},{s}); b=torch.randn({s}); v=torch.randn({s},{v}); torch.softmax(q@kt*0.125+b,dim=-1)@v",
       lambda q, kt, b, v: [(softmax(dot(q, kt) * 0.125 + f64(b)) @ f64(v)).astype(np.float32)])
