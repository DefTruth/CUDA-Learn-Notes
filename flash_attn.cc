#include <torch/extension.h>

torch::Tensor flash_attn_1_fwd_f32(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attn_2_fwd_f32(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_1_fwd_f32", &flash_attn_1_fwd_f32, "FlashAttention1 forward (f32)");
    m.def("flash_attn_2_fwd_f32", &flash_attn_2_fwd_f32, "FlashAttention2 forward (f32)");
}
