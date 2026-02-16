/***********************************************************************
 * simple_gemm.cu
 * 
 * 使用 CUTE 库实现简单的矩阵乘法 (GEMM) 示例代码。
 * 
 * 矩阵规模：(16, 8, 8)
 * 算子精度：fp16 = fp16 * fp16 + fp16
 * Grid shape: (1, 1, 1)
 * Block shape: (32, 1, 1)
 * Block tile shape: (16, 8, 8)
 * TileMMA shape: (16, 8, 8)
 * MMA Atom shape: (16, 8, 8)
 * 
 * 因为是简单示例，只构建了一个 Block，并且只有一个 warp，只进行了一次 MMA_Atom 运算。
 * 所以 Block tile shape，TileMMA shape，MMA Atom shape 都是 (16, 8, 8)。
 * 
***********************************************************************/

# include <cute/tensor.hpp>
# include <cuda_runtime.h>
# include <iostream>
# include "../common/gemm_utils.cuh"

template <typename spec, bool IsGemm>
__global__ void minimal_gemm_kernel(void* Cptr, const void* Aptr, const void* Bptr,
                                    int m, int n, int k) {
  using namespace cute;

  using T = typename spec::T;
  using TiledMMA = typename spec::TiledMMA;
  constexpr int kTileM = spec::kTileM;
  constexpr int kTileN = spec::kTileN;
  constexpr int kTileK = spec::kTileK;

  Tensor mA = make_tensor(make_gmem_ptr((T*)Aptr),
                          make_shape(m, k),
                          make_stride(k, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr((T*)Bptr),
                          make_shape(n, k),
                          make_stride(k, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr((T*)Cptr),
                          make_shape(m, n),
                          make_stride(n, Int<1>{}));

  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  // 当前只有一个 Block，因此 coord 固定为 （0, 0, 0）
  auto coord = make_coord(0, 0, 0);

  // 定位每个 block 负责的 tile
  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});  // (kTileM, kTileK)
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});  // (kTileN, kTileK)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});  // (kTileM, kTileN)

  // 定位每个线程负责的矩阵元素
  TiledMMA tiled_mma;
  int tid = threadIdx.x;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);
  
  Tensor tCgA = thr_mma.partition_A(gA);
  Tensor tCgB = thr_mma.partition_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = thr_mma.partition_fragment_A(gA);
  Tensor tCrB = thr_mma.partition_fragment_B(gB);
  Tensor tCrC = thr_mma.partition_fragment_C(gC);

  auto copy_atom = AutoVectorizingCopy{};
  copy(copy_atom, tCgA, tCrA);
  copy(copy_atom, tCgB, tCrB);

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  copy(copy_atom, tCrC, tCgC);

  if (thread0()) {
    print_latex(tiled_mma); printf("\n");
    print(gA); printf("\n");
    print_tensor(gA); printf("\n");
  }
  return;
}

template <typename T_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8>
struct KernelSpec {

  using MMA_op = cute::SM80_16x8x8_F16F16F16F16_TN;

  using T = T_;
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using TiledMMA = decltype(cute::make_tiled_mma(MMA_op{}));
  static constexpr int kThreadNum = cute::size(TiledMMA{});
  static constexpr int kShmSize = 0;
};


int main() {
  using namespace cute;
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 8;

  GemmData<cute::half_t> gemm(M, N, K);
  gemm.randomize();

  using spec = KernelSpec<cute::half_t>;
  // std::cout << "kThreadNum: " << spec::kThreadNum << std::endl;
  dim3 block = spec::kThreadNum;
  dim3 grid((M + spec::kTileM - 1) / spec::kTileM, (N + spec::kTileN - 1) / spec::kTileN);
  int shm_size = spec::kShmSize;
  minimal_gemm_kernel<spec, true><<<grid, block, shm_size>>>(gemm.d_C, gemm.d_A, gemm.d_B, M, N, K);

  return 0;
}