#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(32) void k_matmul_79479f(const __half* b, const __half* a, __half* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 32;
        int a1_b = (_gid / 32) % 1;
        int a0_u = (_gid / 32) % 1;
        int a1_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 7; a2++) {
            __half in0__ar0 = a[(a0_b + a0_u) * 7 + a2];
            __half in1__bc0 = b[a2 * 19 + (a1_b * 32 + a1_u) % 19];
            __half v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += __half2float(v0__c0_0);
        }
        if (a1_b * 32 + a1_u * 1 + 0 < 19) {
            matmul[(a0_b + a0_u) * 19 + (a1_b * 32 + a1_u) % 19] = __float2half(acc0__c0_0);
        }
}
