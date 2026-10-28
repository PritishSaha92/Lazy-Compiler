extern "C" __global__
__launch_bounds__(256) void k_mul_1_pointwise(const float* add_c1, const float* x, const float* y, float* mul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 2053) {
        int a0 = _gid;
        float in1 = x[a0];
        float in0 = add_c1[0];
        float in2 = y[a0];
        float v4 = (1.0f / (1.0f + expf(0.0f - in1))) * (in2 / (in0 + (in1 * in1)));
        mul_1[a0] = v4;
    }
}
