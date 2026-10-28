extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const int* y, const int* x, int* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1043) {
        int a0 = _gid;
        int in0 = x[a0];
        int in1 = y[a0];
        bool v1 = in0 < in1;
        int v3 = ((v1) ? ((in0 - in1)) : ((in0 + in1)));
        where[a0] = v3;
    }
}
