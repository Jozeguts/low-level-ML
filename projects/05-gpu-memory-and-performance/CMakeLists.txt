cmake_minimum_required(VERSION 3.18)
project(gpu_memory_performance LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(vector_add_bench vector_add_bench.cu)
