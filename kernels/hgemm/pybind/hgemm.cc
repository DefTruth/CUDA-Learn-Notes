#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

// from hgemm.cu
void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_pack(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
// from hgemm_async.cu
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
// from hgemm_cublas.cu
void init_cublas_handle();
void destroy_cublas_handle();
void hgemm_cublas_tensor_op_nn(torch::Tensor a, torch::Tensor b, torch::Tensor c); 
void hgemm_cublas_tensor_op_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c);
// from hgemm_wmma.cu
void hgemm_wmma_m16n16k16_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
// from hgemm_wmma_stage.cu
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);                                                        
void hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
// from hgemm_mma.cu
void hgemm_mma_m16n8k16_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_mma_m16n8k16_mma2x4_warp4x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
// from hgemm_mma_stage.cu
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
// from hgemm_mma_stage_tn.cu
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
// from hgemm_mma_stage_tn_cute.cu
void hgemm_mma_stages_block_swizzle_tn_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
// from hgemm_mma_stage_swizzle.cu
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CUDA Cores FP16
  TORCH_BINDING_COMMON_EXTENSION(hgemm_naive_f16)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_sliced_k_f16)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k_f16x4)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k_f16x4_pack)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k_f16x4_bcf)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k_f16x4_pack_bcf)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k_f16x8_pack_bcf)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf)
  // Copy Async
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async)
  // cuBLAS Tensor Cores
  TORCH_BINDING_COMMON_EXTENSION(init_cublas_handle)
  TORCH_BINDING_COMMON_EXTENSION(destroy_cublas_handle)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublas_tensor_op_nn)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublas_tensor_op_tn)
  // WMMA API Tensor Cores
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m16n16k16_naive)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m16n16k16_mma4x2)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m16n16k16_mma4x2_warp2x4)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async)
  // stage, thread block swizzle, dsmem
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem)
  // MMA API Tensor Cores
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_naive)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4)
  // stage, thread block swizzle, dsmem, reg double buffers
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr)
  // smem swizzle
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle)
  // TN: A row major MxK, B col major NxK, C row major MxN
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn)
  // TN: cute hgemm with smem & block swizzle
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_stages_block_swizzle_tn_cute)
}

