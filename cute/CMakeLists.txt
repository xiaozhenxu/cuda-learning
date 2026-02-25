project(cute_examples)

add_library(cute_lib INTERFACE)
target_compile_features(cute_lib INTERFACE cxx_std_17)
set(CMAKE_CXX_STANDARD 17)

target_include_directories(cute_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass/include)

add_subdirectory(00_simple_gemm)
add_subdirectory(01_simple_mixed_gemm)
add_subdirectory(02_tiled_mma)
add_subdirectory(03_block_mma)