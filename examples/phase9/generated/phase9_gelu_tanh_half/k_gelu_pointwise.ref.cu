#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_gelu_pointwise(const __half* gelu_c1, const __half* gelu_c0, const __half* gelu_one, const __half* gelu_half, const __half* x, __half* gelu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1031) {
        int a0 = _gid;
        __half in4 = x[a0];
        float v0 = __half2float(in4) * __half2float(in4);
        float v1 = __half2float(in4) * v0;
        __half in1 = gelu_c1[0];
        float v2 = __half2float(in1) * v1;
        float v3 = __half2float(in4) + v2;
        __half in0 = gelu_c0[0];
        float v4 = __half2float(in0) * v3;
        __half in2 = gelu_half[0];
        __half in3 = gelu_one[0];
        float v7 = __half2float(in3) + tanhf(v4);
        float v8 = __half2float((in2 * in4)) * v7;
        gelu[a0] = __float2half(v8);
    }
}
