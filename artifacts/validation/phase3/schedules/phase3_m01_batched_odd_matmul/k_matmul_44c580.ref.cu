extern "C" __global__
__launch_bounds__(128) void k_matmul_44c580(const float* b, const float* a, float* matmul) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 45696;
        int a1 = (_gid / 2688) % 17;
        int a2 = (_gid / 128) % 21;
        int a3_co = (_gid) % 128;
        float acc0 = 0.0f;
        for (int a3 = a3_co; a3 < 33; a3 += 128) {
            float in0 = a[a0 * 561 + a1 * 33 + a3];
            float in1 = b[a0 * 693 + a3 * 21 + a2];
            float v0 = in0 * in1;
            acc0 += v0;
        }
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
        if (a3_co == 0) {
            matmul[a0 * 357 + a1 * 21 + a2] = acc0;
        }
}
