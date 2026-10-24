extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_887429(const float* mul_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 27) {
        int a0_b = _gid / 3;
        int a5_b = (_gid) % 3;
        int a0_u = (_gid) % 1;
        int a5_u = (_gid) % 1;
        float in0 = mul_c1[0];
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        float acc3__c0_2 = 0.0f;
        float acc5__sum__c0_2 = 0.0f;
        float acc1__c0_2 = -1e+30f;
        float acc3__c0_3 = 0.0f;
        float acc5__sum__c0_3 = 0.0f;
        float acc1__c0_3 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        float acc1__c1_1 = -1e+30f;
        float acc3__c1_2 = 0.0f;
        float acc5__sum__c1_2 = 0.0f;
        float acc1__c1_2 = -1e+30f;
        float acc3__c1_3 = 0.0f;
        float acc5__sum__c1_3 = 0.0f;
        float acc1__c1_3 = -1e+30f;
        #pragma unroll
        for (int a1 = 0; a1 < 43; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 23; a2++) {
                float in2__ar0 = q[(a0_b * 2 + a0_u * 2) % 17 * 23 + a2];
                float in1__ar0 = kt[a2 * 43 + a1];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 23; a2++) {
                float in2__ar1 = q[(a0_b * 2 + a0_u * 2 + 1) % 17 * 23 + a2];
                float in1__ar1 = kt[a2 * 43 + a1];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__c0_0 = acc0__ar0 * in0;
            float one__c0_0 = 1.0f;
            float v1__gn__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__alpha__c0_0 = expf((acc1__c0_0 - v1__gn__c0_0));
            float v1__beta__c0_0 = expf((v1__c0_0 - v1__gn__c0_0));
            float v1__acc3_sa__c0_0 = acc3__c0_0 * v1__alpha__c0_0;
            float v1__acc3_sb__c0_0 = one__c0_0 * v1__beta__c0_0;
            acc3__c0_0 = v1__acc3_sa__c0_0 + v1__acc3_sb__c0_0;
            float v1__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v1__alpha__c0_0;
            float in7__bc0 = v[a1 * 11 + (a5_b * 4 + a5_u * 4) % 11];
            float v1__acc5__sum_sb__c0_0 = in7__bc0 * v1__beta__c0_0;
            acc5__sum__c0_0 = v1__acc5__sum_sa__c0_0 + v1__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__c0_1 = acc0__ar0 * in0;
            float one__c0_1 = 1.0f;
            float v1__gn__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__alpha__c0_1 = expf((acc1__c0_1 - v1__gn__c0_1));
            float v1__beta__c0_1 = expf((v1__c0_1 - v1__gn__c0_1));
            float v1__acc3_sa__c0_1 = acc3__c0_1 * v1__alpha__c0_1;
            float v1__acc3_sb__c0_1 = one__c0_1 * v1__beta__c0_1;
            acc3__c0_1 = v1__acc3_sa__c0_1 + v1__acc3_sb__c0_1;
            float v1__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v1__alpha__c0_1;
            float in7__bc1 = v[a1 * 11 + (a5_b * 4 + a5_u * 4 + 1) % 11];
            float v1__acc5__sum_sb__c0_1 = in7__bc1 * v1__beta__c0_1;
            acc5__sum__c0_1 = v1__acc5__sum_sa__c0_1 + v1__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__c0_2 = acc0__ar0 * in0;
            float one__c0_2 = 1.0f;
            float v1__gn__c0_2 = fmaxf(acc1__c0_2, v1__c0_2);
            float v1__alpha__c0_2 = expf((acc1__c0_2 - v1__gn__c0_2));
            float v1__beta__c0_2 = expf((v1__c0_2 - v1__gn__c0_2));
            float v1__acc3_sa__c0_2 = acc3__c0_2 * v1__alpha__c0_2;
            float v1__acc3_sb__c0_2 = one__c0_2 * v1__beta__c0_2;
            acc3__c0_2 = v1__acc3_sa__c0_2 + v1__acc3_sb__c0_2;
            float v1__acc5__sum_sa__c0_2 = acc5__sum__c0_2 * v1__alpha__c0_2;
            float in7__bc2 = v[a1 * 11 + (a5_b * 4 + a5_u * 4 + 2) % 11];
            float v1__acc5__sum_sb__c0_2 = in7__bc2 * v1__beta__c0_2;
            acc5__sum__c0_2 = v1__acc5__sum_sa__c0_2 + v1__acc5__sum_sb__c0_2;
            acc1__c0_2 = fmaxf(acc1__c0_2, v1__c0_2);
            float v1__c0_3 = acc0__ar0 * in0;
            float one__c0_3 = 1.0f;
            float v1__gn__c0_3 = fmaxf(acc1__c0_3, v1__c0_3);
            float v1__alpha__c0_3 = expf((acc1__c0_3 - v1__gn__c0_3));
            float v1__beta__c0_3 = expf((v1__c0_3 - v1__gn__c0_3));
            float v1__acc3_sa__c0_3 = acc3__c0_3 * v1__alpha__c0_3;
            float v1__acc3_sb__c0_3 = one__c0_3 * v1__beta__c0_3;
            acc3__c0_3 = v1__acc3_sa__c0_3 + v1__acc3_sb__c0_3;
            float v1__acc5__sum_sa__c0_3 = acc5__sum__c0_3 * v1__alpha__c0_3;
            float in7__bc3 = v[a1 * 11 + (a5_b * 4 + a5_u * 4 + 3) % 11];
            float v1__acc5__sum_sb__c0_3 = in7__bc3 * v1__beta__c0_3;
            acc5__sum__c0_3 = v1__acc5__sum_sa__c0_3 + v1__acc5__sum_sb__c0_3;
            acc1__c0_3 = fmaxf(acc1__c0_3, v1__c0_3);
            float v1__c1_0 = acc0__ar1 * in0;
            float one__c1_0 = 1.0f;
            float v1__gn__c1_0 = fmaxf(acc1__c1_0, v1__c1_0);
            float v1__alpha__c1_0 = expf((acc1__c1_0 - v1__gn__c1_0));
            float v1__beta__c1_0 = expf((v1__c1_0 - v1__gn__c1_0));
            float v1__acc3_sa__c1_0 = acc3__c1_0 * v1__alpha__c1_0;
            float v1__acc3_sb__c1_0 = one__c1_0 * v1__beta__c1_0;
            acc3__c1_0 = v1__acc3_sa__c1_0 + v1__acc3_sb__c1_0;
            float v1__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v1__alpha__c1_0;
            float v1__acc5__sum_sb__c1_0 = in7__bc0 * v1__beta__c1_0;
            acc5__sum__c1_0 = v1__acc5__sum_sa__c1_0 + v1__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v1__c1_0);
            float v1__c1_1 = acc0__ar1 * in0;
            float one__c1_1 = 1.0f;
            float v1__gn__c1_1 = fmaxf(acc1__c1_1, v1__c1_1);
            float v1__alpha__c1_1 = expf((acc1__c1_1 - v1__gn__c1_1));
            float v1__beta__c1_1 = expf((v1__c1_1 - v1__gn__c1_1));
            float v1__acc3_sa__c1_1 = acc3__c1_1 * v1__alpha__c1_1;
            float v1__acc3_sb__c1_1 = one__c1_1 * v1__beta__c1_1;
            acc3__c1_1 = v1__acc3_sa__c1_1 + v1__acc3_sb__c1_1;
            float v1__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v1__alpha__c1_1;
            float v1__acc5__sum_sb__c1_1 = in7__bc1 * v1__beta__c1_1;
            acc5__sum__c1_1 = v1__acc5__sum_sa__c1_1 + v1__acc5__sum_sb__c1_1;
            acc1__c1_1 = fmaxf(acc1__c1_1, v1__c1_1);
            float v1__c1_2 = acc0__ar1 * in0;
            float one__c1_2 = 1.0f;
            float v1__gn__c1_2 = fmaxf(acc1__c1_2, v1__c1_2);
            float v1__alpha__c1_2 = expf((acc1__c1_2 - v1__gn__c1_2));
            float v1__beta__c1_2 = expf((v1__c1_2 - v1__gn__c1_2));
            float v1__acc3_sa__c1_2 = acc3__c1_2 * v1__alpha__c1_2;
            float v1__acc3_sb__c1_2 = one__c1_2 * v1__beta__c1_2;
            acc3__c1_2 = v1__acc3_sa__c1_2 + v1__acc3_sb__c1_2;
            float v1__acc5__sum_sa__c1_2 = acc5__sum__c1_2 * v1__alpha__c1_2;
            float v1__acc5__sum_sb__c1_2 = in7__bc2 * v1__beta__c1_2;
            acc5__sum__c1_2 = v1__acc5__sum_sa__c1_2 + v1__acc5__sum_sb__c1_2;
            acc1__c1_2 = fmaxf(acc1__c1_2, v1__c1_2);
            float v1__c1_3 = acc0__ar1 * in0;
            float one__c1_3 = 1.0f;
            float v1__gn__c1_3 = fmaxf(acc1__c1_3, v1__c1_3);
            float v1__alpha__c1_3 = expf((acc1__c1_3 - v1__gn__c1_3));
            float v1__beta__c1_3 = expf((v1__c1_3 - v1__gn__c1_3));
            float v1__acc3_sa__c1_3 = acc3__c1_3 * v1__alpha__c1_3;
            float v1__acc3_sb__c1_3 = one__c1_3 * v1__beta__c1_3;
            acc3__c1_3 = v1__acc3_sa__c1_3 + v1__acc3_sb__c1_3;
            float v1__acc5__sum_sa__c1_3 = acc5__sum__c1_3 * v1__alpha__c1_3;
            float v1__acc5__sum_sb__c1_3 = in7__bc3 * v1__beta__c1_3;
            acc5__sum__c1_3 = v1__acc5__sum_sa__c1_3 + v1__acc5__sum_sb__c1_3;
            acc1__c1_3 = fmaxf(acc1__c1_3, v1__c1_3);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a0_b * 2 + a0_u * 2 + 0 < 17 && a5_b * 4 + a5_u * 4 + 0 < 11) {
            matmul_1[(a0_b * 2 + a0_u * 2) % 17 * 11 + (a5_b * 4 + a5_u * 4) % 11] = acc5__c0_0;
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a0_b * 2 + a0_u * 2 + 0 < 17 && a5_b * 4 + a5_u * 4 + 1 < 11) {
            matmul_1[(a0_b * 2 + a0_u * 2) % 17 * 11 + (a5_b * 4 + a5_u * 4 + 1) % 11] = acc5__c0_1;
        }
        float acc5__c0_2 = acc5__sum__c0_2 * (1.0f / acc3__c0_2);
        if (a0_b * 2 + a0_u * 2 + 0 < 17 && a5_b * 4 + a5_u * 4 + 2 < 11) {
            matmul_1[(a0_b * 2 + a0_u * 2) % 17 * 11 + (a5_b * 4 + a5_u * 4 + 2) % 11] = acc5__c0_2;
        }
        float acc5__c0_3 = acc5__sum__c0_3 * (1.0f / acc3__c0_3);
        if (a0_b * 2 + a0_u * 2 + 0 < 17 && a5_b * 4 + a5_u * 4 + 3 < 11) {
            matmul_1[(a0_b * 2 + a0_u * 2) % 17 * 11 + (a5_b * 4 + a5_u * 4 + 3) % 11] = acc5__c0_3;
        }
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        if (a0_b * 2 + a0_u * 2 + 1 < 17 && a5_b * 4 + a5_u * 4 + 0 < 11) {
            matmul_1[(a0_b * 2 + a0_u * 2 + 1) % 17 * 11 + (a5_b * 4 + a5_u * 4) % 11] = acc5__c1_0;
        }
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        if (a0_b * 2 + a0_u * 2 + 1 < 17 && a5_b * 4 + a5_u * 4 + 1 < 11) {
            matmul_1[(a0_b * 2 + a0_u * 2 + 1) % 17 * 11 + (a5_b * 4 + a5_u * 4 + 1) % 11] = acc5__c1_1;
        }
        float acc5__c1_2 = acc5__sum__c1_2 * (1.0f / acc3__c1_2);
        if (a0_b * 2 + a0_u * 2 + 1 < 17 && a5_b * 4 + a5_u * 4 + 2 < 11) {
            matmul_1[(a0_b * 2 + a0_u * 2 + 1) % 17 * 11 + (a5_b * 4 + a5_u * 4 + 2) % 11] = acc5__c1_2;
        }
        float acc5__c1_3 = acc5__sum__c1_3 * (1.0f / acc3__c1_3);
        if (a0_b * 2 + a0_u * 2 + 1 < 17 && a5_b * 4 + a5_u * 4 + 3 < 11) {
            matmul_1[(a0_b * 2 + a0_u * 2 + 1) % 17 * 11 + (a5_b * 4 + a5_u * 4 + 3) % 11] = acc5__c1_3;
        }
    }
}
