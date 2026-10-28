extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 2057) {
        int a0 = _gid;
        float in0 = x[a0];
        float v2 = cosf(in0) + sinf(in0);
        add[a0] = v2;
    }
}
