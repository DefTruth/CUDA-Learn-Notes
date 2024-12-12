#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

void flash_attn_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);
void flash_attn_mma_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);
void flash_attn_mma_stages(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int stages);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_cuda)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_naive)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages)
}
