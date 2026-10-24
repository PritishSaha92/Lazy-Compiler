extern "C" __global__
__launch_bounds__(128) void k_mul_1_reduce(const float* add_3_c1, const float* x, const float* r, const float* w, float* mul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 61; a1 += 128) {
            float in1 = r[a0 * 61 + a1];
            float in2 = x[a0 * 61 + a1];
            float v0 = in1 + in2;
            float v2 = v0 * v0;
            acc0 += v2;
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
        float in0 = add_3_c1[0];
        float v5 = 1.0f / sqrtf((acc0 + in0));
        for (int a2 = a1_co; a2 < 61; a2 += 128) {
            float in3 = r[a0 * 61 + a2];
            float in4 = x[a0 * 61 + a2];
            float in5 = w[a2];
            float v8 = in5 * (v5 * (in3 + in4));
            mul_1[a0 * 61 + a2] = v8;
        }
}
