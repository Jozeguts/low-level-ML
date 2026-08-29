#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void vector_add_kernel(const float* a, const float* b, float* out,
                                  std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(EXIT_FAILURE); } } while (0)

int main() {
    constexpr std::size_t n = 1 << 24;
    constexpr int warmup = 10;
    constexpr int iterations = 100;
    constexpr int threads = 256;
    const std::size_t bytes = n * sizeof(float);
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_out(n);
    float *a, *b, *out;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemcpy(a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) vector_add_kernel<<<blocks, threads>>>(a, b, out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i)
        vector_add_kernel<<<blocks, threads>>>(a, b, out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double kernel_ms = total_ms / iterations;
    const double gb = static_cast<double>(bytes) * 3.0 / 1e9;
    const double bandwidth = gb / (kernel_ms / 1000.0);

    CUDA_CHECK(cudaMemcpy(h_out.data(), out, bytes, cudaMemcpyDeviceToHost));
    if (h_out[n / 2] != 3.0f) return EXIT_FAILURE;

    std::printf("N=%zu\nthreads=%d\nblocks=%d\niterations=%d\n", n, threads, blocks, iterations);
    std::printf("kernel_ms=%.6f\nestimated_effective_bandwidth_GBps=%.3f\n", kernel_ms, bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(out));
}
