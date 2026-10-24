extern "C" __global__
__launch_bounds__(128) void k_softmax_86069b(const float* mul_c1, const float* x, const float* g, float* sum_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float in0 = mul_c1[0];
        float acc1 = 0.0f;
        float acc2__sum = 0.0f;
        float acc0 = -1e+30f;
        for (int a1 = a1_co; a1 < 137; a1 += 128) {
            float in1 = x[a0 * 137 + a1];
            float v0 = in0 * in1;
            float one = 1.0f;
            float v0__gn = fmaxf(acc0, v0);
            float v0__alpha = expf((acc0 - v0__gn));
            float v0__beta = expf((v0 - v0__gn));
            float v0__acc1_sa = acc1 * v0__alpha;
            float v0__acc1_sb = one * v0__beta;
            acc1 = v0__acc1_sa + v0__acc1_sb;
            float v0__acc2__sum_sa = acc2__sum * v0__alpha;
            float in3 = g[a0 * 137 + a1];
            float v0__acc2__sum_sb = in3 * v0__beta;
            acc2__sum = v0__acc2__sum_sa + v0__acc2__sum_sb;
            acc0 = fmaxf(acc0, v0);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            float acc2__sum__o = __shfl_xor_sync(__activemask(), acc2__sum, 16);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            float acc0__o__acc2__sum_sa = acc0__o__alpha * acc2__sum;
            float acc0__o__acc2__sum_sb = acc0__o__beta * acc2__sum__o;
            acc2__sum = acc0__o__acc2__sum_sa + acc0__o__acc2__sum_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            float acc2__sum__o = __shfl_xor_sync(__activemask(), acc2__sum, 8);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            float acc0__o__acc2__sum_sa = acc0__o__alpha * acc2__sum;
            float acc0__o__acc2__sum_sb = acc0__o__beta * acc2__sum__o;
            acc2__sum = acc0__o__acc2__sum_sa + acc0__o__acc2__sum_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            float acc2__sum__o = __shfl_xor_sync(__activemask(), acc2__sum, 4);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            float acc0__o__acc2__sum_sa = acc0__o__alpha * acc2__sum;
            float acc0__o__acc2__sum_sb = acc0__o__beta * acc2__sum__o;
            acc2__sum = acc0__o__acc2__sum_sa + acc0__o__acc2__sum_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            float acc2__sum__o = __shfl_xor_sync(__activemask(), acc2__sum, 2);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            float acc0__o__acc2__sum_sa = acc0__o__alpha * acc2__sum;
            float acc0__o__acc2__sum_sb = acc0__o__beta * acc2__sum__o;
            acc2__sum = acc0__o__acc2__sum_sa + acc0__o__acc2__sum_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            float acc2__sum__o = __shfl_xor_sync(__activemask(), acc2__sum, 1);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            float acc0__o__acc2__sum_sa = acc0__o__alpha * acc2__sum;
            float acc0__o__acc2__sum_sb = acc0__o__beta * acc2__sum__o;
            acc2__sum = acc0__o__acc2__sum_sa + acc0__o__acc2__sum_sb;
            acc0 = acc0__o__gn;
        }
        __shared__ float acc0_smem[4];
        __shared__ float acc1_smem[4];
        __shared__ float acc2__sum_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
            acc1_smem[warp] = acc1;
            acc2__sum_smem[warp] = acc2__sum;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                float acc2__sum = acc2__sum_smem[warp];
                float acc2__sum__o = acc2__sum_smem[warp + s];
                float acc0__o__gn = fmaxf(acc0, acc0__o);
                float acc0__o__dg = acc0 - acc0__o__gn;
                float acc0__o__alpha = expf(acc0__o__dg);
                float acc0__o__dg_o = acc0__o - acc0__o__gn;
                float acc0__o__beta = expf(acc0__o__dg_o);
                float acc0__o__acc1_sa = acc0__o__alpha * acc1;
                float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
                acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
                float acc0__o__acc2__sum_sa = acc0__o__alpha * acc2__sum;
                float acc0__o__acc2__sum_sb = acc0__o__beta * acc2__sum__o;
                acc2__sum = acc0__o__acc2__sum_sa + acc0__o__acc2__sum_sb;
                acc0 = acc0__o__gn;
                acc0_smem[warp] = acc0;
                acc1_smem[warp] = acc1;
                acc2__sum_smem[warp] = acc2__sum;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        acc1 = acc1_smem[0];
        acc2__sum = acc2__sum_smem[0];
        if (a1_co == 0) {
            float acc2 = acc2__sum * (1.0f / acc1);
            sum_1[a0] = acc2;
        }
}
