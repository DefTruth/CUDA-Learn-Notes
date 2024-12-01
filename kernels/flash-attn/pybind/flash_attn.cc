#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

void flash_attn_1_fwd_f32(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);
void flash_attn_2_fwd_f16_mma_m16n8k16(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_1_fwd_f32)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_2_fwd_f16_mma_m16n8k16)
}
