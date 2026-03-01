# include <cuda_runtime.h>

__global__ void naive_sgemm_kernel(const float* __restrict__ Aptr,
                                   const float* __restrict__ Bptr,
                                   float* __restrict__ Cptr,
                                   const int M, const int N, const int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float acc = 0.0;
    for (i = 0; i < K; i++) {
        acc += Aptr[row * K + i] * Bptr[i * N + col];
    }
    Cptr[row * N + col] = acc;
}
