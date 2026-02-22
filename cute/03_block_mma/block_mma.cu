/***********************************************************************
 * block_mma.cu
 * 
 * 使用 CUTE 库实现矩阵乘法 (GEMM) 示例代码
 * 在 M N 维度扩展线程数量，在 K 维度扩展执行 mma 指令次数，最终扩展 TiledMMA shape
 * 在 M N K 维度扩展 TiledMMA 执行次数，最终扩展 Block tile shape
 * 
 * 矩阵规模：(128, 128, 64)
 * 算子精度：FP32 = BF16 * BF16 + FP32
 * Grid shape: (1, 1, 1)
 * Block shape: (256, 1, 1)
 * Block tile shape: (128, 128, 64)
 * TileMMA shape: (32, 32, 32)
 * MMA Atom shape: (16, 8, 16)
 * 
 * 每个 TiledMMA 的执行包括 8 个 warp，在 M 维度排布 2 个 warp，在 N 维度排布 4 个 warp
 * 每个 block 在 M 维度循环执行 4 次，在 N 维度循环执行 4 次，在 K 维度循环执行 2 次的 TiledMMA
 * 仅有 1 个 block
 * 
***********************************************************************/

# include <cute/tensor.hpp>
# include <cuda_runtime.h>
# include "../common/gemm_utils.cuh"

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kTileM_ = 128,    // block tile
          int kTileN_ = 128,
          int kTileK_ = 64>
struct KernelSpec {
    using OutType = OutType_;
    using ComputeTypeA = ComputeTypeA_;
    using ComputeTypeB = ComputeTypeB_;
    using ComputeTypeC = ComputeTypeC_;

    static constexpr kTileM = kTileM_;
    static constexpr kTileN = kTileN_;
    static constexpr kTileK = kTileK_;

    static constexpr int kMmaThrExpandM = 2;
    static constexpr int kMmaThrExpandN = 4;
    static constexpr int kMmaThrExpandK = 1;
       
    static constexpr int kMmaValExpandM = 1;
    static constexpr int kMmaValExpandN = 1;
    static constexpr int kMmaValExpandK = 2;

    using MMA_op = SM80_16x8x16_F32BF16BF16F32_TN;
    using MMA_traits = MMA_Traits<MMA_op>;
    using MMA_atom = MMA_Atom<MMA_traits>;
    using MMA_shape = MMA_traits::Shape_MNK;    // MMA Atom shape

    static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape);
    static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape);
    static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape);

    using MMAThrLayout = decltype(make_layout(make_shape(kMmaThrExpandM, kMmaThrExpandN, kMmaThrExpandK)));
    using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
    using TiledMMA = decltype(make_tiled_mma(MMA_op{},MMAThrLayout{}, MMATileLayout{}));

    using Copy_op = AutoVectorizingCopy;

    using CopyA_atom = Copy_Atom<Copy_op, ComputeTypeA>;
    using CopyB_atom = Copy_Atom<Copy_op, ComputeTypeB>;
    using CopyC_atom = Copy_Atom<Copy_op, ComputeTypeC>;
    using CopyO_atom = Copy_Atom<Copy_op, OutType>;

    using TiledCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
    using TiledCopyB = decltype(make_tiled_copy_B(CopyB_atom{}, TiledMMA{}));
    using TiledCopyC = decltype(make_tiled_copy_C(CopyC_atom{}, TiledMMA{}));
    using TiledCopyO = decltype(make_tiled_copy_C(CopyO_atom{}, TiledMMA{}));

    static constexpr int kThreadNum = size(TiledMMA{});
    static constexpr int kShmSize = 0;
};
 
// 这边得看mma指令计算的精度和out精度是否一样
template<typename spec, bool IsGemm>
__global__ void block_mma_kernel(void* Outptr, void* Aptr, void* Bptr, void* Cptr,
                                 int m, int n, int k) {
    using namespace cute;

    using ComputeTypeA = spec::ComputeTypeA;
    using ComputeTypeB = spec::ComputeTypeB;
    using ComputeTypeC = spec::ComputeTypeC;
    using OutType = spec::OutType;

    static constexpr int kTileM = spec::kTileM;
    static constexpr int kTileN = spec::kTileN;
    static constexpr int kTileK = spec::kTileK;

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
    auto coord = make_coord(0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>);
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>);
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>);
    Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>);

    int tid = threadIdx.x;
    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

#pragma unroll
  for (int m_tile = 0; m_tile < NTilesM; ++m_tile) {
#pragma unroll
    for (int n_tile = 0; n_tile < NTilesN; ++n_tile) {
#pragma unroll
      for (int k_tile = 0; k_tile < NTilesK; ++k_tile) {
        copy(g2r_tiled_copy_a, tAgA(_, m_tile, k_tile), tArA(_, m_tile, k_tile));
        copy(g2r_tiled_copy_b, tBgB(_, n_tile, k_tile), tBrB(_, n_tile, k_tile));
#pragma unroll
        for (int im = m_tile * kMmaValExpandM; im < (m_tile + 1) * kMmaValExpandM; ++im) {
#pragma unroll
          for (int in = n_tile * kMmaValExpandN; in < (n_tile + 1) * kMmaValExpandN; ++in) {
#pragma unroll
            for (int ik = k_tile * kMmaValExpandK; ik < (k_tile + 1) * kMmaValExpandK; ++ik) {
              gemm(tiled_mma, tCrC(_, im, in), tCrA(_, im, ik), tCrB(_, in, ik), tCrC(_, im, in));
            }
          }
        }
      }
    }
  } 
};

int main() {
    using namespace cute;

    using OutType = float;
    using ComputeTypeA = half_t;
    using ComputeTypeB = half_t;
    using ComputeTypeC = float;
    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 64;

    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 64;
    GemmData<ComputeTypeA, ComputeTypeB, ComputeTypeC, OutType>(M, N, K);
    
    using Spec = KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, kTileM, kTileN, kTileK>;
    dim3 block = Spec::kThreadNum;
    dim3 grid((M + Spec::kTileM - 1) / Spec::kTileM, (N + Spec::kTileN - 1) / Spec::kTileN);
    int shm_size = Spec::kShmSize;
    block_mma_kernel<spec, bool><<<grid, block, shm_size>>>;
    return 0;
}