extern "C" __global__
__launch_bounds__(256) void k_gelu_pointwise(const float* gelu_c1, const float* gelu_c0, const float* gelu_one, const float* gelu_half, const float* x, float* gelu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 2603) {
        int a0 = _gid;
        float in4 = x[a0];
        float in1 = gelu_c1[0];
        float in0 = gelu_c0[0];
        float in2 = gelu_half[0];
        float in3 = gelu_one[0];
        float v8 = (in2 * in4) * (in3 + tanhf((in0 * (in4 + (in1 * (in4 * (in4 * in4)))))));
        gelu[a0] = v8;
    }
}
