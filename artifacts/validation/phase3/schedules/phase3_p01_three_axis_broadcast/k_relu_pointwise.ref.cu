extern "C" __global__
__launch_bounds__(256) void k_relu_pointwise(const float* y, const float* x, const float* z, float* relu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 819) {
        int a0 = _gid / 117;
        int a1 = (_gid / 13) % 9;
        int a2 = (_gid) % 13;
        float in1 = x[a0 * 13 + a2];
        float in0 = y[a1];
        float in2 = z[a0 * 117 + a1 * 13 + a2];
        float v2 = fmaxf(0.0f, ((in0 + in1) - in2));
        relu[a0 * 117 + a1 * 13 + a2] = v2;
    }
}
