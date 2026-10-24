extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_396951(const float* mul_c1, const float* mul_1_c1, const float* q, const float* kt, const float* r, const float* v, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 60) {
        int a0_b = _gid / 10;
        int a5_b = (_gid) % 10;
        int a0_u = (_gid) % 1;
        int a5_u = (_gid) % 1;
        float in1 = mul_c1[0];
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        float acc3__c0_2 = 0.0f;
        float acc5__sum__c0_2 = 0.0f;
        float acc1__c0_2 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        float acc1__c1_1 = -1e+30f;
        float acc3__c1_2 = 0.0f;
        float acc5__sum__c1_2 = 0.0f;
        float acc1__c1_2 = -1e+30f;
        for (int a1 = 0; a1 < 80; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 20; a2++) {
                float in3__ar0 = q[(a0_b * 2 + a0_u * 2) * 20 + a2];
                float in2__ar0 = kt[a2 * 80 + a1];
                float v0__ar0 = in2__ar0 * in3__ar0;
                acc0__ar0 += v0__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 20; a2++) {
                float in3__ar1 = q[(a0_b * 2 + a0_u * 2 + 1) * 20 + a2];
                float in2__ar1 = kt[a2 * 80 + a1];
                float v0__ar1 = in2__ar1 * in3__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__c0_0 = acc0__ar0 * in1;
            float one__c0_0 = 1.0f;
            float v1__gn__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__alpha__c0_0 = expf((acc1__c0_0 - v1__gn__c0_0));
            float v1__beta__c0_0 = expf((v1__c0_0 - v1__gn__c0_0));
            float v1__acc3_sa__c0_0 = acc3__c0_0 * v1__alpha__c0_0;
            float v1__acc3_sb__c0_0 = one__c0_0 * v1__beta__c0_0;
            acc3__c0_0 = v1__acc3_sa__c0_0 + v1__acc3_sb__c0_0;
            float v1__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v1__alpha__c0_0;
            float in8__bc0 = v[a1 * 28 + (a5_b * 3 + a5_u * 3) % 28];
            float v1__acc5__sum_sb__c0_0 = in8__bc0 * v1__beta__c0_0;
            acc5__sum__c0_0 = v1__acc5__sum_sa__c0_0 + v1__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__c0_1 = acc0__ar0 * in1;
            float one__c0_1 = 1.0f;
            float v1__gn__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__alpha__c0_1 = expf((acc1__c0_1 - v1__gn__c0_1));
            float v1__beta__c0_1 = expf((v1__c0_1 - v1__gn__c0_1));
            float v1__acc3_sa__c0_1 = acc3__c0_1 * v1__alpha__c0_1;
            float v1__acc3_sb__c0_1 = one__c0_1 * v1__beta__c0_1;
            acc3__c0_1 = v1__acc3_sa__c0_1 + v1__acc3_sb__c0_1;
            float v1__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v1__alpha__c0_1;
            float in8__bc1 = v[a1 * 28 + (a5_b * 3 + a5_u * 3 + 1) % 28];
            float v1__acc5__sum_sb__c0_1 = in8__bc1 * v1__beta__c0_1;
            acc5__sum__c0_1 = v1__acc5__sum_sa__c0_1 + v1__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__c0_2 = acc0__ar0 * in1;
            float one__c0_2 = 1.0f;
            float v1__gn__c0_2 = fmaxf(acc1__c0_2, v1__c0_2);
            float v1__alpha__c0_2 = expf((acc1__c0_2 - v1__gn__c0_2));
            float v1__beta__c0_2 = expf((v1__c0_2 - v1__gn__c0_2));
            float v1__acc3_sa__c0_2 = acc3__c0_2 * v1__alpha__c0_2;
            float v1__acc3_sb__c0_2 = one__c0_2 * v1__beta__c0_2;
            acc3__c0_2 = v1__acc3_sa__c0_2 + v1__acc3_sb__c0_2;
            float v1__acc5__sum_sa__c0_2 = acc5__sum__c0_2 * v1__alpha__c0_2;
            float in8__bc2 = v[a1 * 28 + (a5_b * 3 + a5_u * 3 + 2) % 28];
            float v1__acc5__sum_sb__c0_2 = in8__bc2 * v1__beta__c0_2;
            acc5__sum__c0_2 = v1__acc5__sum_sa__c0_2 + v1__acc5__sum_sb__c0_2;
            acc1__c0_2 = fmaxf(acc1__c0_2, v1__c0_2);
            float v1__c1_0 = acc0__ar1 * in1;
            float one__c1_0 = 1.0f;
            float v1__gn__c1_0 = fmaxf(acc1__c1_0, v1__c1_0);
            float v1__alpha__c1_0 = expf((acc1__c1_0 - v1__gn__c1_0));
            float v1__beta__c1_0 = expf((v1__c1_0 - v1__gn__c1_0));
            float v1__acc3_sa__c1_0 = acc3__c1_0 * v1__alpha__c1_0;
            float v1__acc3_sb__c1_0 = one__c1_0 * v1__beta__c1_0;
            acc3__c1_0 = v1__acc3_sa__c1_0 + v1__acc3_sb__c1_0;
            float v1__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v1__alpha__c1_0;
            float v1__acc5__sum_sb__c1_0 = in8__bc0 * v1__beta__c1_0;
            acc5__sum__c1_0 = v1__acc5__sum_sa__c1_0 + v1__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v1__c1_0);
            float v1__c1_1 = acc0__ar1 * in1;
            float one__c1_1 = 1.0f;
            float v1__gn__c1_1 = fmaxf(acc1__c1_1, v1__c1_1);
            float v1__alpha__c1_1 = expf((acc1__c1_1 - v1__gn__c1_1));
            float v1__beta__c1_1 = expf((v1__c1_1 - v1__gn__c1_1));
            float v1__acc3_sa__c1_1 = acc3__c1_1 * v1__alpha__c1_1;
            float v1__acc3_sb__c1_1 = one__c1_1 * v1__beta__c1_1;
            acc3__c1_1 = v1__acc3_sa__c1_1 + v1__acc3_sb__c1_1;
            float v1__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v1__alpha__c1_1;
            float v1__acc5__sum_sb__c1_1 = in8__bc1 * v1__beta__c1_1;
            acc5__sum__c1_1 = v1__acc5__sum_sa__c1_1 + v1__acc5__sum_sb__c1_1;
            acc1__c1_1 = fmaxf(acc1__c1_1, v1__c1_1);
            float v1__c1_2 = acc0__ar1 * in1;
            float one__c1_2 = 1.0f;
            float v1__gn__c1_2 = fmaxf(acc1__c1_2, v1__c1_2);
            float v1__alpha__c1_2 = expf((acc1__c1_2 - v1__gn__c1_2));
            float v1__beta__c1_2 = expf((v1__c1_2 - v1__gn__c1_2));
            float v1__acc3_sa__c1_2 = acc3__c1_2 * v1__alpha__c1_2;
            float v1__acc3_sb__c1_2 = one__c1_2 * v1__beta__c1_2;
            acc3__c1_2 = v1__acc3_sa__c1_2 + v1__acc3_sb__c1_2;
            float v1__acc5__sum_sa__c1_2 = acc5__sum__c1_2 * v1__alpha__c1_2;
            float v1__acc5__sum_sb__c1_2 = in8__bc2 * v1__beta__c1_2;
            acc5__sum__c1_2 = v1__acc5__sum_sa__c1_2 + v1__acc5__sum_sb__c1_2;
            acc1__c1_2 = fmaxf(acc1__c1_2, v1__c1_2);
        }
        float in0__c0_0 = mul_1_c1[0];
        float in9__c0_0 = r[(a0_b * 2 + a0_u * 2) * 28 + (a5_b * 3 + a5_u * 3) % 28];
        float v14__c0_0 = (acc5__sum__c0_0 * (1.0f / acc3__c0_0)) + (in0__c0_0 * in9__c0_0);
        if (a5_b * 3 + a5_u * 3 + 0 < 28) {
            add[(a0_b * 2 + a0_u * 2) * 28 + (a5_b * 3 + a5_u * 3) % 28] = v14__c0_0;
        }
        float in0__c0_1 = mul_1_c1[0];
        float in9__c0_1 = r[(a0_b * 2 + a0_u * 2) * 28 + (a5_b * 3 + a5_u * 3 + 1) % 28];
        float v14__c0_1 = (acc5__sum__c0_1 * (1.0f / acc3__c0_1)) + (in0__c0_1 * in9__c0_1);
        if (a5_b * 3 + a5_u * 3 + 1 < 28) {
            add[(a0_b * 2 + a0_u * 2) * 28 + (a5_b * 3 + a5_u * 3 + 1) % 28] = v14__c0_1;
        }
        float in0__c0_2 = mul_1_c1[0];
        float in9__c0_2 = r[(a0_b * 2 + a0_u * 2) * 28 + (a5_b * 3 + a5_u * 3 + 2) % 28];
        float v14__c0_2 = (acc5__sum__c0_2 * (1.0f / acc3__c0_2)) + (in0__c0_2 * in9__c0_2);
        if (a5_b * 3 + a5_u * 3 + 2 < 28) {
            add[(a0_b * 2 + a0_u * 2) * 28 + (a5_b * 3 + a5_u * 3 + 2) % 28] = v14__c0_2;
        }
        float in0__c1_0 = mul_1_c1[0];
        float in9__c1_0 = r[(a0_b * 2 + a0_u * 2 + 1) * 28 + (a5_b * 3 + a5_u * 3) % 28];
        float v14__c1_0 = (acc5__sum__c1_0 * (1.0f / acc3__c1_0)) + (in0__c1_0 * in9__c1_0);
        if (a5_b * 3 + a5_u * 3 + 0 < 28) {
            add[(a0_b * 2 + a0_u * 2 + 1) * 28 + (a5_b * 3 + a5_u * 3) % 28] = v14__c1_0;
        }
        float in0__c1_1 = mul_1_c1[0];
        float in9__c1_1 = r[(a0_b * 2 + a0_u * 2 + 1) * 28 + (a5_b * 3 + a5_u * 3 + 1) % 28];
        float v14__c1_1 = (acc5__sum__c1_1 * (1.0f / acc3__c1_1)) + (in0__c1_1 * in9__c1_1);
        if (a5_b * 3 + a5_u * 3 + 1 < 28) {
            add[(a0_b * 2 + a0_u * 2 + 1) * 28 + (a5_b * 3 + a5_u * 3 + 1) % 28] = v14__c1_1;
        }
        float in0__c1_2 = mul_1_c1[0];
        float in9__c1_2 = r[(a0_b * 2 + a0_u * 2 + 1) * 28 + (a5_b * 3 + a5_u * 3 + 2) % 28];
        float v14__c1_2 = (acc5__sum__c1_2 * (1.0f / acc3__c1_2)) + (in0__c1_2 * in9__c1_2);
        if (a5_b * 3 + a5_u * 3 + 2 < 28) {
            add[(a0_b * 2 + a0_u * 2 + 1) * 28 + (a5_b * 3 + a5_u * 3 + 2) % 28] = v14__c1_2;
        }
    }
}
