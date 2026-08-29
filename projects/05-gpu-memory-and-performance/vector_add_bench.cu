#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(EXIT_FAILURE); } } while (0)

__global__ void vector_add(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

static double benchmark(int threads, const float* a, const float* b, float* out, std::size_t n) {
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    for (int i = 0; i < 10; ++i) vector_add<<<blocks, threads>>>(a, b, out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i) vector_add<<<blocks, threads>>>(a, b, out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / 100.0;
}

int main() {
    constexpr std::size_t n = 1 << 24;
    constexpr std::size_t bytes = n * sizeof(float);
    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_out(n);
    float *a, *b, *out;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemcpy(a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    std::printf("threads,blocks,kernel_ms,effective_GBps\n");
    for (int threads : {32, 64, 128, 256, 512, 1024}) {
        const int blocks = static_cast<int>((n + threads - 1) / threads);
        const double ms = benchmark(threads, a, b, out, n);
        const double bandwidth = (3.0 * static_cast<double>(bytes) / 1e9) / (ms / 1000.0);
        std::printf("%d,%d,%.6f,%.3f\n", threads, blocks, ms, bandwidth);
    }

    CUDA_CHECK(cudaMemcpy(h_out.data(), out, bytes, cudaMemcpyDeviceToHost));
    if (h_out[n / 2] != 3.0f) return EXIT_FAILURE;
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(out));
}
