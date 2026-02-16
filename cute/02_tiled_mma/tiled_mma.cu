/***********************************************************************
 * tiled_mma.cu
 * 
 * 使用 CUTE 库实现矩阵乘法 (GEMM) 示例代码
 * 在 M N 维度扩展线程数量，在 K 维度扩展执行 mma 指令次数，最终扩展 TiledMMA shape
 * 
 * 矩阵规模：(32, 32, 16)
 * 算子精度：BF16 = BF16 * BF16 + FP32
 * Grid shape: (1, 1, 1)
 * Block shape: (256, 1, 1)
 * Block tile shape: (32, 32, 16)
 * TileMMA shape: (32, 32, 16)
 * MMA Atom shape: (16, 8, 8)
 * 
 * 仅构建了一个 Block，每个 Block 包括 8 个 warp，在 M 维度排布 2 个 warp，在 N 维度排布 4 个 warp。
 * 
***********************************************************************/

# include <cute/tensor.hpp>
# include <cuda_runtime.h>
# include <iostream>
# include "../common/gemm_utils.cuh"

template <typename spec, bool IsGemm, bool IsCvtPrecision>
__global__ void tiled_mma_kernel(void* Outptr, void* Aptr, void* Bptr, void* Cptr,
                                 int m, int n, int k) {
    using namespace cute;

    using OutType = typename spec::OutType;
    using ComputeTypeA = typename spec::ComputeTypeA;
    using ComputeTypeB = typename spec::ComputeTypeB;
    using ComputeTypeC = typename spec::ComputeTypeC;
    using TiledMMA = typename spec::TiledMMA;

    constexpr int kTileM = spec::kTileM;
    constexpr int kTileN = spec::kTileN;
    constexpr int kTileK = spec::kTileK;

    Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA*)Aptr),
                                           make_shape(m, k),
                                           make_stride(k, 1));
    Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB*)Bptr),
                                           make_shape(n, k),
                                           make_stride(k, 1));
    Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC*)Cptr),
                                           make_shape(m, n),
                                           make_stride(n, 1));
    Tensor mO = make_tensor(make_gmem_ptr((OutType*)Outptr),
                                           make_shape(m, n),
                                           make_stride(n, 1));

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    // Define the block global tensors (static)
    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (kTileM, kTileK)
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (kTileN, kTileK)
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)
    Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)

    TiledMMA tiled_mma;
    int tid = threadIdx.x;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    Tensor tCgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    Tensor tCrA = thr_mma.partition_fragment_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB); // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(gC); // (MMA, MMA_M, MMA_N)

    auto copy_atom = AutoVectorizingCopy{};

    copy(copy_atom, tCgA, tCrA);
    copy(copy_atom, tCgB, tCrB);
    if constexpr (IsGemm)
        clear(tCrC); // Set the accumulators to zero
    else
        copy(copy_atom, tCgC, tCrC);

    // 当前在 gemm 计算的时候都是将 tCrC 同时作为输出地址
    // todo：后续可以统一一下架构
    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    if constexpr (!IsCvtPrecision) {
        copy(copy_atom, tCrC, tCgC);
    } else {
        Tensor tCgO = thr_mma.partition_C(gO); // (MMA, MMA_M, MMA_N)
        auto t = make_tensor_like<OutType>(tCrC);
        copy(tCrC, t); // Convert precision
        copy(copy_atom, t, tCgO);
    }
}

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kTileM_ = 32,  // block shape
          int kTileN_ = 32,
          int kTileK_ = 16>
struct KernelSpec {
    using OutType = OutType_;
    using ComputeTypeA = ComputeTypeA_;
    using ComputeTypeB = ComputeTypeB_;
    using ComputeTypeC = ComputeTypeC_;

    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;

    using MMA_op = cute::SM80_16x8x8_F32BF16BF16F32_TN;
    using MMA_traits = cute::MMA_Traits<MMA_op>;
    using MMA_atom = cute::MMA_Atom<MMA_traits>;
    using MMA_shape = MMA_traits::Shape_MNK;    // 对应 MMA Atom Shape

    // 线程扩展
    static constexpr int kMmaThrExpandM = 2;
    static constexpr int kMmaThrExpandN = 4;
    static constexpr int kMmaThrExpandK = 1;

    // 指令扩展
    static constexpr int kMmaValExpandM = 1;
    static constexpr int kMmaValExpandN = 1;
    static constexpr int kMmaValExpandK = 2;

    // Tiled MMA Shape
    static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * cute::get<0>(MMA_shape{});
    static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * cute::get<1>(MMA_shape{});
    static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * cute::get<2>(MMA_shape{});

    using MMAThrLayout = decltype(cute::make_layout(cute::make_shape(cute::Int<kMmaThrExpandM>{}, cute::Int<kMmaThrExpandN>{}, cute::Int<kMmaThrExpandK>{})));
    using MMATileLayout = cute::Tile<cute::Int<kMmaTileM>, cute::Int<kMmaTileN>, cute::Int<kMmaTileK>>;

    using TiledMMA = decltype(cute::make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

    static constexpr int kThreadNum = cute::size(TiledMMA{});
    static constexpr int kShmSize = 0;
};

int main() {
  using namespace cute;
  constexpr int M = 32;
  constexpr int N = 32;
  constexpr int K = 16;

  GemmData<cute::half_t> gemm(M, N, K);
  gemm.randomize();

  // todo: 后续将 spec 统一一下
  using spec = KernelSpec<cute::half_t, cute::half_t, cute::half_t, float, M, N, K>;
  // std::cout << "kThreadNum: " << spec::kThreadNum << std::endl;
  dim3 block = spec::kThreadNum;
  dim3 grid((M + spec::kTileM - 1) / spec::kTileM, (N + spec::kTileN - 1) / spec::kTileN);
  int shm_size = spec::kShmSize;
  tiled_mma_kernel<spec, true, true><<<grid, block, shm_size>>>(gemm.d_C, gemm.d_A, gemm.d_B, gemm.d_C, M, N, K);

  return 0;
}