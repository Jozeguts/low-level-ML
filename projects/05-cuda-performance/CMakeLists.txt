cmake_minimum_required(VERSION 3.18)
project(cuda_performance LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(vector_add_benchmark day-005/benchmark.cu)
