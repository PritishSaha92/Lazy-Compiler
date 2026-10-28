extern "C" __global__
__launch_bounds__(128) void k_sum_1_reduce(const float* gelu_inv_sqrt2, const float* gelu_one, const float* gelu_half, const float* x, const float* g, float* sum_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float in0 = gelu_half[0];
        float in1 = gelu_inv_sqrt2[0];
        float in2 = gelu_one[0];
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 131; a1 += 128) {
            float in4 = x[a0 * 131 + a1];
            float in3 = g[a0 * 131 + a1];
            float v5 = in3 * ((in2 + erff((in1 * in4))) * (in0 * in4));
            acc0 += v5;
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
        if (a1_co == 0) {
            sum_1[a0] = acc0;
        }
}
