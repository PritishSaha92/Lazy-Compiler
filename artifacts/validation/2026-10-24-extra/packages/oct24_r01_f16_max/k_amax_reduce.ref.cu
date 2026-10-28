#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(128) void k_amax_reduce(const __half* x, __half* amax) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        __half acc0 = __float2half(-1e+30f);
        for (int a1 = a1_co; a1 < 257; a1 += 128) {
            __half in0 = x[a0 * 257 + a1];
            acc0 = __hmax(acc0, in0);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        __shared__ float acc0_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = __half2float(acc0);
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = fmaxf(acc0, acc0__o);
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        if (a1_co == 0) {
            __half v0 = acc0;
            amax[a0] = v0;
        }
}
