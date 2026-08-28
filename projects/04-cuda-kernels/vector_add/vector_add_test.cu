#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void vector_add(const float*, const float*, float*, std::size_t);

static void check_case(std::size_t n) {
    std::vector<float> a(n), b(n), out(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.25f;
        b[i] = static_cast<float>(i % 17) - 3.0f;
    }

    vector_add(a.data(), b.data(), out.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (std::fabs(out[i] - expected) > 1e-6f) {
            std::fprintf(stderr, "mismatch at %zu: got %.8f expected %.8f\n",
                         i, out[i], expected);
            std::exit(EXIT_FAILURE);
        }
    }
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA device available\n");
        return EXIT_FAILURE;
    }

    for (std::size_t n : {1, 31, 256, 257, 1000, 1 << 20}) {
        check_case(n);
    }

    std::puts("All CUDA vector-add tests passed.");
    return EXIT_SUCCESS;
}
