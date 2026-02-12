cmake_minimum_required(VERSION 3.30)

project(cuda_learning LANGUAGES CXX CUDA)

# Ampere 3090
set(CMAKE_CUDA_ARCHITECTURES 80)

add_subdirectory(cute)