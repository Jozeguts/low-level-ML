#include <torch/extension.h>

#include <cstdint>

namespace lowlevel {

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cpu(), "a must be a CPU tensor");
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    auto out = torch::empty_like(a);
    const auto n = a.numel();
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    return out;
}

}  // namespace lowlevel

TORCH_LIBRARY(lowlevel_ops, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(lowlevel_ops, CPU, m) {
    m.impl("vector_add", &lowlevel::vector_add);
}
