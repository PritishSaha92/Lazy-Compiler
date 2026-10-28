extern "C" __global__
__launch_bounds__(128) void k_mean_reduce_bfe0aa(const double* mean_2_count, const double* mean_1_count, const double* mean_count, const double* add_c1, const double* x, double* div) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        for (int a1 = a1_co; a1 < 71; a1 += 128) {
            float in4 = x[a0 * 71 + a1];
            acc0 += in4;
            acc1 += in4;
            acc2 += in4;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 16);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 8);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 4);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 2);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 1);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        __shared__ float acc0_smem[4];
        __shared__ float acc1_smem[4];
        __shared__ float acc2_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
            acc1_smem[warp] = acc1;
            acc2_smem[warp] = acc2;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                float acc2 = acc2_smem[warp];
                float acc2__o = acc2_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc1 = acc1 + acc1__o;
                acc2 = acc2 + acc2__o;
                acc0_smem[warp] = acc0;
                acc1_smem[warp] = acc1;
                acc2_smem[warp] = acc2;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        acc1 = acc1_smem[0];
        acc2 = acc2_smem[0];
        float in2 = mean_2_count[0];
        float v1 = 1.0f / in2;
        float v3 = acc1 * v1;
        float in1 = mean_1_count[0];
        float v0 = 1.0f / in1;
        float v4 = acc0 * v0;
        float acc3 = 0.0f;
        for (int a1 = 0; a1 < 71; a1++) {
            float in5 = x[a0 * 71 + a1];
            float v5 = in5 - v3;
            float v6 = in5 - v4;
            float v7 = v5 * v6;
            acc3 += v7;
        }
        float v8 = acc3;
        float in0 = add_c1[0];
        float v12 = 1.0f / sqrtf((in0 + v8));
        float in3 = mean_count[0];
        float v2 = 1.0f / in3;
        float v10 = acc2 * v2;
        for (int a3 = a1_co; a3 < 71; a3 += 128) {
            float in6 = x[a0 * 71 + a3];
            float v13 = in6 - v10;
            float v14 = v12 * v13;
            div[a0 * 71 + a3] = v14;
        }
}
