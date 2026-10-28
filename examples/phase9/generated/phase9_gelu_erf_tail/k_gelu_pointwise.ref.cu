extern "C" __global__
__launch_bounds__(256) void k_gelu_pointwise(const float* gelu_inv_sqrt2, const float* gelu_one, const float* gelu_half, const float* x, float* gelu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 4099) {
        int a0 = _gid;
        float in1 = gelu_inv_sqrt2[0];
        float in3 = x[a0];
        float in2 = gelu_one[0];
        float in0 = gelu_half[0];
        float v4 = (in2 + erff((in1 * in3))) * (in0 * in3);
        gelu[a0] = v4;
    }
}
