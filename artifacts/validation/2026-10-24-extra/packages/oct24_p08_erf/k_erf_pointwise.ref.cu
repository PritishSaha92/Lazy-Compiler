extern "C" __global__
__launch_bounds__(256) void k_erf_pointwise(const float* x, float* erf) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1073) {
        int a0 = _gid;
        float in0 = x[a0];
        float v0 = erff(in0);
        erf[a0] = v0;
    }
}
