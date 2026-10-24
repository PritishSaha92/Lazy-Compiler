extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* y, const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        float in0 = x[a0];
        float in1 = y[a0];
        float v1 = in1 * (in0 - in1);
        mul[a0] = v1;
    }
}
