extern "C" __global__
__launch_bounds__(256) void k_tanh_pointwise(const float* b, const float* w, const float* x, float* tanh) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1081) {
        int a0 = _gid / 47;
        int a1 = (_gid) % 47;
        float in2 = w[a1];
        float in1 = x[a0 * 47 + a1];
        float in0 = b[a0];
        float v2 = tanhf((in0 + (in1 * in2)));
        tanh[a0 * 47 + a1] = v2;
    }
}
