# include <cuda_runtime.h>

template <const int BM, const int BN, const int BK>
__global__ void sgemm_shared_memory_kernel(const float* __restrict__ Aptr,
                                           const float* __restrict__ Bptr,
                                           float* __restrict__ Cptr,
                                           const int M, const int N, const int K) {
    __shared__ float block_tileA[BM * BK];
    __shared__ float block_tileB[BK * BN];
}