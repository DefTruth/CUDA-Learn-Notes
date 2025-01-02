#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// Basic
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

void flash_attn_mma_stages_split_q_tiling_qk(torch::Tensor Q, 
                                             torch::Tensor K, 
                                             torch::Tensor V, 
                                             torch::Tensor O, 
                                             int stages);

// HMMA F32F16F16F32 acc with F32 dtype.
void flash_attn_mma_stages_split_q_shared_kv_acc_f32(torch::Tensor Q, 
                                                     torch::Tensor K, 
                                                     torch::Tensor V, 
                                                     torch::Tensor O, 
                                                     int stages);

void flash_attn_mma_stages_split_q_shared_qkv_acc_f32(torch::Tensor Q, 
                                                      torch::Tensor K, 
                                                      torch::Tensor V, 
                                                      torch::Tensor O, 
                                                      int stages);      

void flash_attn_mma_stages_split_q_tiling_qk_acc_f32(torch::Tensor Q, 
                                                     torch::Tensor K, 
                                                     torch::Tensor V, 
                                                     torch::Tensor O, 
                                                     int stages);

// Swizzle
// shared memory swizzle for Q, K, V
void flash_attn_mma_stages_split_q_shared_kv_swizzle_q(torch::Tensor Q, 
                                                       torch::Tensor K, 
                                                       torch::Tensor V, 
                                                       torch::Tensor O, 
                                                       int stages);

void flash_attn_mma_stages_split_q_shared_kv_swizzle_qk(torch::Tensor Q, 
                                                        torch::Tensor K, 
                                                        torch::Tensor V, 
                                                        torch::Tensor O, 
                                                        int stages);

void flash_attn_mma_stages_split_q_shared_kv_swizzle_qkv(torch::Tensor Q, 
                                                         torch::Tensor K, 
                                                         torch::Tensor V, 
                                                         torch::Tensor O, 
                                                         int stages);

void flash_attn_mma_stages_split_q_shared_qkv_swizzle_q(torch::Tensor Q, 
                                                        torch::Tensor K, 
                                                        torch::Tensor V, 
                                                        torch::Tensor O, 
                                                        int stages);

void flash_attn_mma_stages_split_q_shared_qkv_swizzle_qk(torch::Tensor Q, 
                                                         torch::Tensor K, 
                                                         torch::Tensor V, 
                                                         torch::Tensor O, 
                                                         int stages);

void flash_attn_mma_stages_split_q_shared_qkv_swizzle_qkv(torch::Tensor Q, 
                                                          torch::Tensor K, 
                                                          torch::Tensor V, 
                                                          torch::Tensor O, 
                                                          int stages);

void flash_attn_mma_stages_split_q_tiling_qk_swizzle_q(torch::Tensor Q, 
                                                       torch::Tensor K, 
                                                       torch::Tensor V, 
                                                       torch::Tensor O, 
                                                       int stages);

void flash_attn_mma_stages_split_q_tiling_qk_swizzle_qk(torch::Tensor Q, 
                                                        torch::Tensor K, 
                                                        torch::Tensor V, 
                                                        torch::Tensor O, 
                                                        int stages);

void flash_attn_mma_stages_split_q_tiling_qk_swizzle_qkv(torch::Tensor Q, 
                                                         torch::Tensor K, 
                                                         torch::Tensor V, 
                                                         torch::Tensor O, 
                                                         int stages);

// Others
#ifdef BUILD_FLASH_ATTN_MMA_OTHERS
// O collective store using shared memory, O s2g.
void flash_attn_mma_stages_split_q_shared_qkv_Os2g(torch::Tensor Q, 
                                                   torch::Tensor K, 
                                                   torch::Tensor V, 
                                                   torch::Tensor O, 
                                                   int stages);  
// reduce registers usage.
void flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr(torch::Tensor Q, 
                                                         torch::Tensor K, 
                                                         torch::Tensor V, 
                                                         torch::Tensor O, 
                                                         int stages);

void flash_attn_mma_stages_split_q_shared_kv_acc_f32_rr(torch::Tensor Q, 
                                                        torch::Tensor K, 
                                                        torch::Tensor V, 
                                                        torch::Tensor O, 
                                                        int stages);

void flash_attn_mma_stages_split_q_tiling_qk_acc_f32_rr(torch::Tensor Q, 
                                                        torch::Tensor K, 
                                                        torch::Tensor V, 
                                                        torch::Tensor O, 
                                                        int stages);
#endif 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Basic
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_kv)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_kv)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_qkv)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_tiling_qk)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_kv_acc_f32)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_qkv_acc_f32)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_tiling_qk_acc_f32)
  // Swizzle
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_kv_swizzle_q)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_kv_swizzle_qk)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_kv_swizzle_qkv)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_qkv_swizzle_q)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_qkv_swizzle_qk)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_qkv_swizzle_qkv)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_tiling_qk_swizzle_q)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_tiling_qk_swizzle_qk)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_tiling_qk_swizzle_qkv)
  // Others
#ifdef BUILD_FLASH_ATTN_MMA_OTHERS
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_qkv_Os2g)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_kv_acc_f32_rr)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr)
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_tiling_qk_acc_f32_rr)
#endif
}
