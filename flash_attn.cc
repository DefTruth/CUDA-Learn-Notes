#include <torch/extension.h>

torch::Tensor custom_flash_attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attn_fwd", torch::wrap_pybind_function(custom_flash_attn_fwd), "custom_flash_attn_fwd");
}
