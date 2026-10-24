extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const double* add_1_c1, const double* add_c1, const double* x, double* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1033) {
        int a0 = _gid;
        float in2 = x[a0];
        float in0 = add_1_c1[0];
        float in1 = add_c1[0];
        float v4 = (in1 + (in2 * in2)) / (in0 + fabsf(in2));
        div[a0] = v4;
    }
}
