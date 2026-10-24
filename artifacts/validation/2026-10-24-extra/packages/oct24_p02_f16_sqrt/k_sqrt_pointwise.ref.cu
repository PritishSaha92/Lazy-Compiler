#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_sqrt_pointwise(const __half* add_c1, const __half* x, __half* sqrt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1031) {
        int a0 = _gid;
        __half in1 = x[a0];
        __half in0 = add_c1[0];
        __half v2 = hsqrt((in0 + fabsf(in1)));
        sqrt[a0] = v2;
    }
}
