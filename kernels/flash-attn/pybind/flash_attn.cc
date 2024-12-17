#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

void flash_attn_mma_stages_split_kv(torch::Tensor Q, 
                                    torch::Tensor K, 
                                    torch::Tensor V, 
                                    torch::Tensor O, 
                                    int stages);

void flash_attn_mma_stages_split_q(torch::Tensor Q, 
                                   torch::Tensor K, 
                                   torch::Tensor V, 
                                   torch::Tensor O, 
                                   int stages);

void flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, 
                                             torch::Tensor K, 
                                             torch::Tensor V, 
                                             torch::Tensor O, 
                                             int stages);

void flash_attn_mma_stages_split_q_shared_qkv(torch::Tensor Q, 
                                              torch::Tensor K, 
                                              torch::Tensor V, 
                                              torch::Tensor O, 
                                              int stages);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_kv)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_kv)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_qkv)
}
