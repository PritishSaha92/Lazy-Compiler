extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_cd8d6f(const float* mul_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 256;
        int a5_blk = (_gid / 256) % 1;
        int a1_co = (_gid / 32) % 8;
        int a5_ln = (_gid) % 32;
        float in0 = mul_c1[0];
        float acc3 = 0.0f;
        float acc5__sum = 0.0f;
        float acc1 = -1e+30f;
        for (int a1 = a1_co; a1 < 67; a1 += 8) {
            float acc0 = 0.0f;
            for (int a2 = 0; a2 < 37; a2++) {
                float in2 = q[a0 * 37 + a2];
                float in1 = kt[a2 * 67 + a1];
                float v0 = in1 * in2;
                acc0 += v0;
            }
            float v1 = acc0 * in0;
            float one = 1.0f;
            float v1__gn = fmaxf(acc1, v1);
            float v1__alpha = expf((acc1 - v1__gn));
            float v1__beta = expf((v1 - v1__gn));
            float v1__acc3_sa = acc3 * v1__alpha;
            float v1__acc3_sb = one * v1__beta;
            acc3 = v1__acc3_sa + v1__acc3_sb;
            float v1__acc5__sum_sa = acc5__sum * v1__alpha;
            float in7 = v[a1 * 15 + ((a5_blk * 32 + a5_ln < 15) ? (a5_blk * 32 + a5_ln) : (14))];
            float v1__acc5__sum_sb = in7 * v1__beta;
            acc5__sum = v1__acc5__sum_sa + v1__acc5__sum_sb;
            acc1 = fmaxf(acc1, v1);
        }
        __shared__ float acc1_smem[256];
        __shared__ float acc3_smem[256];
        __shared__ float acc5__sum_smem[256];
        acc1_smem[a1_co * 32 + a5_ln] = acc1;
        acc3_smem[a1_co * 32 + a5_ln] = acc3;
        acc5__sum_smem[a1_co * 32 + a5_ln] = acc5__sum;
        __syncthreads();
        for (int s = 4; s > 0; s >>= 1) {
            if (a1_co < s) {
                float acc1 = acc1_smem[a1_co * 32 + a5_ln];
                float acc1__o = acc1_smem[(a1_co + s) * 32 + a5_ln];
                float acc3 = acc3_smem[a1_co * 32 + a5_ln];
                float acc3__o = acc3_smem[(a1_co + s) * 32 + a5_ln];
                float acc5__sum = acc5__sum_smem[a1_co * 32 + a5_ln];
                float acc5__sum__o = acc5__sum_smem[(a1_co + s) * 32 + a5_ln];
                float acc1__o__gn = fmaxf(acc1, acc1__o);
                float acc1__o__dg = acc1 - acc1__o__gn;
                float acc1__o__alpha = expf(acc1__o__dg);
                float acc1__o__dg_o = acc1__o - acc1__o__gn;
                float acc1__o__beta = expf(acc1__o__dg_o);
                float acc1__o__acc3_sa = acc1__o__alpha * acc3;
                float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
                acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
                float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
                float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
                acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
                acc1 = acc1__o__gn;
                acc1_smem[a1_co * 32 + a5_ln] = acc1;
                acc3_smem[a1_co * 32 + a5_ln] = acc3;
                acc5__sum_smem[a1_co * 32 + a5_ln] = acc5__sum;
            }
            __syncthreads();
        }
        acc1 = acc1_smem[a5_ln];
        acc3 = acc3_smem[a5_ln];
        acc5__sum = acc5__sum_smem[a5_ln];
        if (a1_co == 0) {
            if (a5_blk * 32 + a5_ln < 15) {
                float acc5 = acc5__sum * (1.0f / acc3);
                matmul_1[a0 * 15 + (a5_blk * 32 + a5_ln)] = acc5;
            }
        }
}
