# pragma once
# include <cuda_runtime.h>
# include <iostream>
# include <random>

template <typename ComputeTypeA,
          typename ComputeTypeB,
          typename ComputeTypeC,
          typename ComputeTypeD>
class GemmData {
public:
    ComputeTypeA *d_A = nullptr;
    ComputeTypeB *d_B = nullptr;
    ComputeTypeC *d_C = nullptr;
    ComputeTypeD *d_D = nullptr;
    int M, N, K;

    GemmData(int m, int n, int k) : M(m), N(n), K(k) {
        cudaMalloc((void**)&d_A, M * K * sizeof(ComputeTypeA));
        cudaMalloc((void**)&d_B, K * N * sizeof(ComputeTypeB));
        cudaMalloc((void**)&d_C, M * N * sizeof(ComputeTypeC));
        cudaMalloc((void**)&d_D, M * N * sizeof(ComputeTypeD));
    }

    ~GemmData() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (d_D) cudaFree(d_D);
    }

    GemmData(const GemmData&) =  delete;
    GemmData& operator=(const GemmData&) = delete;

    void randomize(int seed = -1) {
        fillRandom(d_A, M * K, seed);
        fillRandom(d_B, K * N, seed + 1);
        cudaMemset(d_C, 0, M * N * sizeof(ComputeTypeC));
        cudaMemset(d_D, 0, M * N * sizeof(ComputeTypeD));
    }
private:
    // 为不同的类型提供不同的随机数生成策略
    template <typename U>
    void fillRandom(U *d_ptr, size_t count, int seed) {
        std::vector<U> h_data(count);
        std::mt19937 gen(seed == -1 ? std::random_device{}() : seed);

        // 判断 U 是否为整数类型 (int, short, etc.)
        if constexpr (std::is_integral<U>::value) {
            std::uniform_int_distribution<int> dis(-5, 5);
            for (size_t i = 0; i < count; ++i) {
                h_data[i] = static_cast<U>(dis(gen));
            }
        } 
        else {
            // 对于 float, double, cute::half_t, __half 都走这里
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (size_t i = 0; i < count; ++i) {
                float val = dis(gen);
                h_data[i] = static_cast<U>(val);
            }
        }

        cudaMemcpy(d_ptr, h_data.data(), count * sizeof(U), cudaMemcpyHostToDevice);
    }

};