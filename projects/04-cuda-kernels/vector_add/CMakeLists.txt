cmake_minimum_required(VERSION 3.18)
project(vector_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(vector_add STATIC vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(vector_add_test vector_add_test.cu)
target_link_libraries(vector_add_test PRIVATE vector_add)

add_executable(vector_add_benchmark benchmark.cu)
