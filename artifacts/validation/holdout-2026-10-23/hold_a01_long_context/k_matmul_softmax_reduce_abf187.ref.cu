extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_abf187(const float* div_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 48) {
        int a0_b = _gid / 12;
        int a5_b = (_gid) % 12;
        int a0_u = (_gid) % 1;
        int a5_u = (_gid) % 1;
        float in0 = div_c1[0];
        float v1 = 1.0f / in0;
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        for (int a1 = 0; a1 < 192; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 24; a2++) {
                float in2__ar0 = q[(a0_b * 2 + a0_u * 2) * 24 + a2];
                float in1__ar0 = kt[a2 * 192 + a1];
                float v3__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v3__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 24; a2++) {
                float in2__ar1 = q[(a0_b * 2 + a0_u * 2 + 1) * 24 + a2];
                float in1__ar1 = kt[a2 * 192 + a1];
                float v3__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v3__ar1;
            }
            float v4__c0_0 = acc0__ar0 * v1;
            float one__c0_0 = 1.0f;
            float v4__gn__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__alpha__c0_0 = expf((acc1__c0_0 - v4__gn__c0_0));
            float v4__beta__c0_0 = expf((v4__c0_0 - v4__gn__c0_0));
            float v4__acc3_sa__c0_0 = acc3__c0_0 * v4__alpha__c0_0;
            float v4__acc3_sb__c0_0 = one__c0_0 * v4__beta__c0_0;
            acc3__c0_0 = v4__acc3_sa__c0_0 + v4__acc3_sb__c0_0;
            float v4__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v4__alpha__c0_0;
            float in7__bc0 = v[a1 * 12 + (a5_b + a5_u)];
            float v4__acc5__sum_sb__c0_0 = in7__bc0 * v4__beta__c0_0;
            acc5__sum__c0_0 = v4__acc5__sum_sa__c0_0 + v4__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__c1_0 = acc0__ar1 * v1;
            float one__c1_0 = 1.0f;
            float v4__gn__c1_0 = fmaxf(acc1__c1_0, v4__c1_0);
            float v4__alpha__c1_0 = expf((acc1__c1_0 - v4__gn__c1_0));
            float v4__beta__c1_0 = expf((v4__c1_0 - v4__gn__c1_0));
            float v4__acc3_sa__c1_0 = acc3__c1_0 * v4__alpha__c1_0;
            float v4__acc3_sb__c1_0 = one__c1_0 * v4__beta__c1_0;
            acc3__c1_0 = v4__acc3_sa__c1_0 + v4__acc3_sb__c1_0;
            float v4__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v4__alpha__c1_0;
            float v4__acc5__sum_sb__c1_0 = in7__bc0 * v4__beta__c1_0;
            acc5__sum__c1_0 = v4__acc5__sum_sa__c1_0 + v4__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v4__c1_0);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        matmul_1[(a0_b * 2 + a0_u * 2) * 12 + (a5_b + a5_u)] = acc5__c0_0;
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        matmul_1[(a0_b * 2 + a0_u * 2 + 1) * 12 + (a5_b + a5_u)] = acc5__c1_0;
    }
}
