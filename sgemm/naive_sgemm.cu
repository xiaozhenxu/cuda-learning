# include <cuda_runtime.h>

template <const int M = 128, const int N = 128, const K = 128>
__global__ void naive_sgemm_kernel(const void *Aptr, const void *Bptr, void *Cptr) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float acc = 0.0;
    for (i = 0; i < K; i++) {
        acc += Aptr[] * Bptr[];
    }
    
}
