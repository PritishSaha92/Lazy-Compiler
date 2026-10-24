extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1105) {
        int a0 = _gid;
        float in0 = x[a0];
        float v2 = (1.0f / (1.0f + expf(0.0f - in0))) + tanhf(in0);
        add[a0] = v2;
    }
}
