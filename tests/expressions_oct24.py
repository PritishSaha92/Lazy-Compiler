"""Fresh typed-storage and tail cases, fixed before GPU feedback on 24 October.

These expressions are only test-data inputs to tools/generate.py. NumPy
references are independent of both Emmy's renderer and the submitted compiler.
The generator uses seeded normal inputs even for randint InputOp declarations;
the preparation script feeds those same arrays into CPU PyTorch explicitly.
"""

import numpy as np

REDUCE = "WORK=t128,REDUCE=coop"


def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def half_relu(x, y):
    return [np.maximum(((x - y).astype(np.float16) * y).astype(np.float16), np.float16(0))]


OPEN = [
    dict(id="oct24_p01_f16_broadcast", category="pointwise",
         family="FP16 buffer loads/stores, broadcast and explicit stage rounding",
         code="x=torch.randn(11,67,dtype=torch.float16); y=torch.randn(67,dtype=torch.float16); torch.relu((x-y)*y)",
         ref=half_relu, rtol=0, atol=0),
    dict(id="oct24_p02_f16_sqrt", category="pointwise",
         family="FP16 odd tail, absolute value and square root",
         code="x=torch.randn(1031,dtype=torch.float16); torch.sqrt(torch.abs(x)+0.5)",
         ref=lambda x: [np.sqrt((np.abs(x) + np.float16(0.5)).astype(np.float16)).astype(np.float16)],
         rtol=1e-3, atol=1e-3),
    dict(id="oct24_p03_f64_rounding", category="pointwise",
         family="FP64 real buffers with floor/ceil/trunc and ragged row strides",
         code="x=torch.randn(13,79,dtype=torch.float64); torch.floor(x)+torch.ceil(x)-torch.trunc(x)",
         ref=lambda x: [np.floor(x) + np.ceil(x) - np.trunc(x)], rtol=0, atol=0),
    dict(id="oct24_p04_f64_rational", category="pointwise",
         family="FP64 precision-sensitive multiply/divide over an odd tail",
         code="x=torch.randn(1033,dtype=torch.float64); (x*x+0.5)/(torch.abs(x)+1.0)",
         ref=lambda x: [(x*x + 0.5)/(np.abs(x) + 1.0)], rtol=2e-14, atol=2e-14),
    dict(id="oct24_p05_i32_bitwise", category="pointwise",
         family="I32 signed bitwise xor/and with odd two-dimensional layout",
         code="x=torch.randint(-9,10,(17,31),dtype=torch.int32); y=torch.randint(-9,10,(17,31),dtype=torch.int32); torch.bitwise_xor(x,y)+torch.bitwise_and(x,y)",
         ref=lambda x,y: [np.bitwise_xor(x,y)+np.bitwise_and(x,y)], rtol=0, atol=0),
    dict(id="oct24_p06_i32_selection", category="pointwise",
         family="I32 signed comparison and where selection",
         code="x=torch.randint(-9,10,(7,149),dtype=torch.int32); y=torch.randint(-9,10,(7,149),dtype=torch.int32); torch.where(x<y,x-y,x+y)",
         ref=lambda x,y: [np.where(x<y,x-y,x+y)], rtol=0, atol=0),
    dict(id="oct24_p07_sin_cos", category="pointwise",
         family="composed trigonometric operations absent from the public pointwise set",
         code="x=torch.randn(2057); torch.sin(x)+torch.cos(x)",
         ref=lambda x: [np.sin(x)+np.cos(x)], rtol=3e-6, atol=3e-6),
    dict(id="oct24_p08_erf", category="pointwise",
         family="error function over ragged rows",
         code="x=torch.randn(29,37); torch.erf(x)",
         ref=lambda x: [np.vectorize(__import__('math').erf,otypes=[np.float64])(x).astype(np.float32)],
         rtol=3e-6, atol=5e-7),
    dict(id="oct24_p09_remainder", category="pointwise",
         family="floating remainder with negative numerators and odd tail",
         code="x=torch.randn(1049); torch.remainder(x,2.5)",
         ref=lambda x: [np.remainder(x,np.float32(2.5))], rtol=1e-6, atol=1e-6),
    dict(id="oct24_r01_f16_max", category="reduction",
         family="FP16 buffers and ragged cooperative maximum",
         code="torch.amax(torch.randn(19,257,dtype=torch.float16),dim=-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=REDUCE),
    dict(id="oct24_r02_f16_to_f32_sum", category="reduction",
         family="FP16 storage expanded before FP32 cooperative accumulation",
         code="x=torch.randn(13,129,dtype=torch.float16); x.to(torch.float32).sum(-1)",
         ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)],
         rtol=1e-4, atol=1e-4, knobs=REDUCE),
    dict(id="oct24_r03_f64_column_sum", category="reduction",
         family="FP64 strided loads and 64-bit cross-lane state over a ragged axis",
         code="torch.randn(131,17,dtype=torch.float64).sum(0)",
         ref=lambda x: [x.sum(0,dtype=np.float64)], rtol=2e-13, atol=2e-13, knobs=REDUCE),
    dict(id="oct24_r04_i32_column_sum", category="reduction",
         family="I32 inputs with I64 sum state on a non-contiguous axis",
         code="torch.randint(-9,10,(139,23),dtype=torch.int32).sum(0)",
         ref=lambda x: [x.sum(0,dtype=np.int64)], rtol=0, atol=0, knobs=REDUCE),
    dict(id="oct24_m01_f64_ragged", category="matmul",
         family="FP64 typed contraction with three non-divisible dimensions",
         code="a=torch.randn(13,29,dtype=torch.float64); b=torch.randn(29,17,dtype=torch.float64); a@b",
         ref=lambda a,b: [a@b], rtol=1e-12, atol=1e-12),
    dict(id="oct24_m02_f16_ragged", category="matmul",
         family="FP16 typed contraction with short odd K and ragged outputs",
         code="a=torch.randn(11,7,dtype=torch.float16); b=torch.randn(7,19,dtype=torch.float16); a@b",
         ref=lambda a,b: [(a.astype(np.float64)@b.astype(np.float64)).astype(np.float16)],
         rtol=1e-2, atol=1e-2),
    dict(id="oct24_f01_f64_centered_norm", category="fused",
         family="FP64 reduction reuse, centering and square root",
         code="x=torch.randn(11,71,dtype=torch.float64); z=x-x.mean(-1,keepdim=True); z/torch.sqrt((z*z).sum(-1,keepdim=True)+0.01)",
         ref=lambda x: [(x-x.mean(-1,keepdims=True))/np.sqrt(((x-x.mean(-1,keepdims=True))**2).sum(-1,keepdims=True)+0.01)],
         rtol=2e-12, atol=2e-12),
    dict(id="oct24_f02_softmax_tail", category="fused",
         family="coupled normalization beyond a 512-element row boundary",
         code="x=torch.randn(5,521); torch.softmax(x*0.25,dim=-1)",
         ref=lambda x: [softmax(x*np.float32(0.25)).astype(np.float32)], rtol=1e-3, atol=1e-5),
    dict(id="oct24_a01_attention_tail", category="attention",
         family="nested online normalization with all four dimensions odd",
         code="q=torch.randn(9,37); kt=torch.randn(37,67); v=torch.randn(67,15); torch.softmax(q@kt*0.125,dim=-1)@v",
         ref=lambda q,kt,v: [(softmax((q.astype(np.float64)@kt.astype(np.float64))*0.125)@v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
]
