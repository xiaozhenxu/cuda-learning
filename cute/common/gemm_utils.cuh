# pragma once
# include <cuda_runtime.h>
# include <iostream>
# include <random>

template <typename T>
class GemmData {
public:
    T *d_A = nullptr;
    T *d_B = nullptr;
    T *d_C = nullptr;
    int M, N, K;

    GemmData(int m, int n, int k) : M(m), N(n), K(k) {
        cudaMalloc((void**)&d_A, M * K * sizeof(T));
        cudaMalloc((void**)&d_B, K * N * sizeof(T));
        cudaMalloc((void**)&d_C, M * N * sizeof(T));
    }

    ~GemmData() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
    }

    GemmData(const GemmData&) =  delete;
    GemmData& operator=(const GemmData&) = delete;

    void randomize(int seed = -1) {
        fillRandom(d_A, M * K, seed);
        fillRandom(d_B, K * N, seed + 1);
        cudaMemset(d_C, 0, M * N * sizeof(T));
    }
private:
    void fillRandom(T *d_ptr, size_t count, int seed) {
        std::vector<T> h_data(count);
        std::mt19937 gen(seed == -1 ? std::random_device{}() : seed);

        // 判断 T 是否为整数类型 (int, short, etc.)
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<int> dis(-5, 5);
            for (size_t i = 0; i < count; ++i) {
                h_data[i] = static_cast<T>(dis(gen));
            }
        } 
        else {
            // 对于 float, double, cute::half_t, __half 都走这里
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (size_t i = 0; i < count; ++i) {
                float val = dis(gen);
                h_data[i] = static_cast<T>(val);
            }
        }

        cudaMemcpy(d_ptr, h_data.data(), count * sizeof(T), cudaMemcpyHostToDevice);
    }

};