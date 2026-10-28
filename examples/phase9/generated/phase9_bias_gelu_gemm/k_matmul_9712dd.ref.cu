extern "C" __global__
__launch_bounds__(256) void k_matmul_9712dd(const float* gelu_inv_sqrt2, const float* gelu_half, const float* gelu_one, const float* bias, const float* b, const float* a, float* gelu) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 12032;
        int a1 = (_gid / 256) % 47;
        int a2_co = (_gid) % 256;
        float acc0 = 0.0f;
        for (int a2 = a2_co; a2 < 65; a2 += 256) {
            float in3 = a[a0 * 65 + a2];
            float in4 = b[a2 * 47 + a1];
            float v0 = in3 * in4;
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
        __shared__ float acc0_smem[8];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }
        __syncthreads();
        for (int s = 4; s > 0; s >>= 1) {
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
            float in5 = bias[a1];
            float v1 = acc0 + in5;
            float in0 = gelu_half[0];
            float in1 = gelu_inv_sqrt2[0];
            float in2 = gelu_one[0];
            float v6 = (in2 + erff((in1 * v1))) * (in0 * v1);
            gelu[a0 * 47 + a1] = v6;
        }
}
