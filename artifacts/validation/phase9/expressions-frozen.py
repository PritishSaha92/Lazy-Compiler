"""Original GELU composition workloads, fixed before generation or GPU feedback.

Development input generation only. The compiler never imports this module.
References use Python erf and NumPy float64 independently of the PTX emitter.
"""
import math
import numpy as np


def gelu(x, approximate=False):
    x = np.asarray(x, dtype=np.float64)
    if approximate:
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    return 0.5 * x * (1 + np.vectorize(math.erf, otypes=[np.float64])(x / math.sqrt(2)))


def dot(a, b):
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def attention(q, kt, v):
    scores = dot(q, kt) * 0.125
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (weights / weights.sum(axis=-1, keepdims=True)) @ np.asarray(v, dtype=np.float64)


OPEN = [
    dict(id="phase9_gelu_erf_tail", category="pointwise", family="gelu_erf",
         code="x=torch.randn(4099); torch.nn.functional.gelu(x)",
         ref=lambda x: [gelu(x).astype(np.float32)], rtol=2e-5, atol=2e-6),
    dict(id="phase9_gelu_tanh_matrix", category="pointwise", family="gelu_tanh",
         code="x=torch.randn(19,137); torch.nn.functional.gelu(x,approximate='tanh')",
         ref=lambda x: [gelu(x, True).astype(np.float32)], rtol=2e-5, atol=2e-6),
    dict(id="phase9_gelu_erf_half", category="pointwise", family="gelu_half",
         code="x=torch.randn(7,259,dtype=torch.float16); torch.nn.functional.gelu(x)",
         ref=lambda x: [gelu(x).astype(np.float16)], rtol=2e-3, atol=2e-4),
    dict(id="phase9_gelu_tanh_half", category="pointwise", family="gelu_half_tanh",
         code="x=torch.randn(1031,dtype=torch.float16); torch.nn.functional.gelu(x,approximate='tanh')",
         ref=lambda x: [gelu(x, True).astype(np.float16)], rtol=2e-3, atol=2e-4),
    dict(id="phase9_bias_gelu_gemm", category="matmul", family="gelu_gemm_epilogue",
         code="a=torch.randn(33,65); b=torch.randn(65,47); bias=torch.randn(47); torch.nn.functional.gelu(a@b+bias)",
         ref=lambda a,b,bias: [gelu(dot(a,b)+bias).astype(np.float32)], rtol=1e-3, atol=1e-4),
    dict(id="phase9_half_gelu_gemm", category="matmul", family="half_gelu_gemm_epilogue",
         code="a=torch.randn(17,33,dtype=torch.float16); b=torch.randn(33,21,dtype=torch.float16); torch.nn.functional.gelu(a@b,approximate='tanh')",
         ref=lambda a,b: [gelu(dot(a,b).astype(np.float16),True).astype(np.float16)], rtol=3e-3, atol=3e-3),
    dict(id="phase9_gelu_row_dot", category="fused", family="gelu_before_reduction",
         code="x=torch.randn(19,131); g=torch.randn(19,131); (torch.nn.functional.gelu(x)*g).sum(-1)",
         ref=lambda x,g: [(gelu(x)*g).sum(axis=-1).astype(np.float32)], rtol=1e-3, atol=1e-4,
         knobs="WORK=t128,REDUCE=coop"),
    dict(id="phase9_gelu_attention", category="attention", family="gelu_after_attention",
         code="q=torch.randn(11,17); kt=torch.randn(17,37); v=torch.randn(37,13); torch.nn.functional.gelu(torch.softmax(q@kt*0.125,dim=-1)@v)",
         ref=lambda q,kt,v: [gelu(attention(q,kt,v)).astype(np.float32)], rtol=1e-3, atol=1e-4),
]
