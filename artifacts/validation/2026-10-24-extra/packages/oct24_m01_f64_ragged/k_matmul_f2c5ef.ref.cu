extern "C" __global__
__launch_bounds__(128) void k_matmul_f2c5ef(const double* b, const double* a, double* matmul) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 2176;
        int a1 = (_gid / 128) % 17;
        int a2_co = (_gid) % 128;
        float acc0 = 0.0f;
        float acc0__r1 = 0.0f;
        for (int a2 = a2_co; a2 < 29; a2 += 256) {
            float in0 = a[a0 * 29 + a2];
            float in1 = b[a2 * 17 + a1];
            float v0 = in0 * in1;
            acc0 += v0;
            float in0__r1 = a[a0 * 29 + (a2 + 128) % 29];
            float in1__r1 = b[(a2 + 128) % 29 * 17 + a1];
            float v0__r1 = in0__r1 * in1__r1;
            float v0__r1__id = 0.0f;
            float v0__r1__m = ((a2 + 128 < 29) ? (((float)(v0__r1))) : (((float)(v0__r1__id))));
            acc0__r1 += v0__r1__m;
        }
        acc0 += acc0__r1;
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }
        __shared__ float acc0_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        if (a2_co == 0) {
            matmul[a0 * 17 + a1] = acc0;
        }
}
