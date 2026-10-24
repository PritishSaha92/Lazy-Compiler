"""Equivalent inline repair of setup-variable capture, fixed before its GPU test.

The original f01 package remains archived: Emmy captured precomputed z as input.
This spelling keeps x as the input and the centering inside the traced program.
The mathematical NumPy reference and tolerance are unchanged.
"""
import numpy as np

OPEN = [dict(
    id="oct24_f01_f64_centered_norm_inline", category="fused",
    family="FP64 reduction reuse, centering and square root; inline trace repair",
    code="x=torch.randn(11,71,dtype=torch.float64); (x-x.mean(-1,keepdim=True))/torch.sqrt(((x-x.mean(-1,keepdim=True))*(x-x.mean(-1,keepdim=True))).sum(-1,keepdim=True)+0.01)",
    ref=lambda x: [(x-x.mean(-1,keepdims=True))/np.sqrt(((x-x.mean(-1,keepdims=True))**2).sum(-1,keepdims=True)+0.01)],
    rtol=2e-12, atol=2e-12,
)]
