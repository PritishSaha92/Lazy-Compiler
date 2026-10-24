#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_relu_pointwise(const __half* y, const __half* x, __half* relu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 737) {
        int a0 = _gid / 67;
        int a1 = (_gid) % 67;
        __half in0 = x[a0 * 67 + a1];
        __half in1 = y[a1];
        __half v2 = __hmax(__float2half(0.0f), (in1 * (in0 - in1)));
        relu[a0 * 67 + a1] = v2;
    }
}
