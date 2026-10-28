#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_gelu_pointwise(const __half* gelu_inv_sqrt2, const __half* gelu_one, const __half* gelu_half, const __half* x, __half* gelu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1813) {
        int a0 = _gid;
        __half in1 = gelu_inv_sqrt2[0];
        __half in3 = x[a0];
        __half in2 = gelu_one[0];
        __half in0 = gelu_half[0];
        __half v4 = (in2 + erff((in1 * in3))) * (in0 * in3);
        gelu[a0] = v4;
    }
}
