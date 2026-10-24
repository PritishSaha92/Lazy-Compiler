extern "C" __global__
__launch_bounds__(128) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 768;
        int a0_b = (_gid / 384) % 2;
        int a1_b = (_gid / 128) % 3;
        int a0_u = (_gid / 16) % 8;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 3; a2++) {
            float in0__ar0 = x0[(a0_b * 32 + a0_u * 4) * 96 + (_a2_ks * 3 + a2)];
            float in1__bc0 = x1[(_a2_ks * 3 + a2) * 80 + (a1_b * 32 + a1_u * 2) % 80];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in1__bc1 = x1[(_a2_ks * 3 + a2) * 80 + (a1_b * 32 + a1_u * 2 + 1) % 80];
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
            float in0__ar1 = x0[(a0_b * 32 + a0_u * 4 + 1) * 96 + (_a2_ks * 3 + a2)];
            float v0__c1_0 = in0__ar1 * in1__bc0;
            acc0__c1_0 += v0__c1_0;
            float v0__c1_1 = in0__ar1 * in1__bc1;
            acc0__c1_1 += v0__c1_1;
            float in0__ar2 = x0[(a0_b * 32 + a0_u * 4 + 2) * 96 + (_a2_ks * 3 + a2)];
            float v0__c2_0 = in0__ar2 * in1__bc0;
            acc0__c2_0 += v0__c2_0;
            float v0__c2_1 = in0__ar2 * in1__bc1;
            acc0__c2_1 += v0__c2_1;
            float in0__ar3 = x0[(a0_b * 32 + a0_u * 4 + 3) * 96 + (_a2_ks * 3 + a2)];
            float v0__c3_0 = in0__ar3 * in1__bc0;
            acc0__c3_0 += v0__c3_0;
            float v0__c3_1 = in0__ar3 * in1__bc1;
            acc0__c3_1 += v0__c3_1;
        }
        if (a1_b * 32 + a1_u * 2 + 0 < 80) {
            atomicAdd(&matmul[(a0_b * 32 + a0_u * 4) * 80 + (a1_b * 32 + a1_u * 2) % 80], acc0__c0_0);
        }
        if (a1_b * 32 + a1_u * 2 + 1 < 80) {
            atomicAdd(&matmul[(a0_b * 32 + a0_u * 4) * 80 + (a1_b * 32 + a1_u * 2 + 1) % 80], acc0__c0_1);
        }
        if (a1_b * 32 + a1_u * 2 + 0 < 80) {
            atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 1) * 80 + (a1_b * 32 + a1_u * 2) % 80], acc0__c1_0);
        }
        if (a1_b * 32 + a1_u * 2 + 1 < 80) {
            atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 1) * 80 + (a1_b * 32 + a1_u * 2 + 1) % 80], acc0__c1_1);
        }
        if (a1_b * 32 + a1_u * 2 + 0 < 80) {
            atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 2) * 80 + (a1_b * 32 + a1_u * 2) % 80], acc0__c2_0);
        }
        if (a1_b * 32 + a1_u * 2 + 1 < 80) {
            atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 2) * 80 + (a1_b * 32 + a1_u * 2 + 1) % 80], acc0__c2_1);
        }
        if (a1_b * 32 + a1_u * 2 + 0 < 80) {
            atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 3) * 80 + (a1_b * 32 + a1_u * 2) % 80], acc0__c3_0);
        }
        if (a1_b * 32 + a1_u * 2 + 1 < 80) {
            atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 3) * 80 + (a1_b * 32 + a1_u * 2 + 1) % 80], acc0__c3_1);
        }
}
