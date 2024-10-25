#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void nms_kernel(const float *boxes, const float *scores, int *keep, int num_boxes, float iou_threshold) {
  const int threadsPerBlock = blockDim.x;
  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;
  const int idx = blockId * threadsPerBlock + threadId;

  if (idx >= num_boxes)
    return;

  float x1 = boxes[idx * 4 + 0];
  float y1 = boxes[idx * 4 + 1];
  float x2 = boxes[idx * 4 + 2];
  float y2 = boxes[idx * 4 + 3];
  int suppressed = 0;

  for (int i = 0; i < idx; ++i) {
    if (keep[i] == 0)
      continue;

    float x1_i = boxes[i * 4 + 0];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];

    float inter_x1 = max(x1, x1_i);
    float inter_y1 = max(y1, y1_i);
    float inter_x2 = min(x2, x2_i);
    float inter_y2 = min(y2, y2_i);
    float inter_w = max(0.0f, inter_x2 - inter_x1);
    float inter_h = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area = (x2 - x1) * (y2 - y1);
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);
    float iou = inter_area / (area + area_i - inter_area);

    if (iou > iou_threshold) {
      keep[idx] = 0;
      return;
    }
  }
  keep[idx] = 1;
  return;
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type))) {                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

torch::Tensor nms(torch::Tensor boxes, torch::Tensor scores, float iou_threshold) {
  CHECK_TORCH_TENSOR_DTYPE(boxes, torch::kFloat32);
  CHECK_TORCH_TENSOR_DTYPE(scores, torch::kFloat32);
  const int num_boxes = boxes.size(0);
  auto toption = torch::TensorOptions().dtype(torch::kInt32).device(boxes.device());
  auto keep = torch::empty({boxes.size(0)}, toption);
  dim3 block(WARP_SIZE);
  dim3 grid((num_boxes + WARP_SIZE - 1) / WARP_SIZE);
  // sort boxes by scores
  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t).contiguous();
  
  nms_kernel<<<grid, block>>>(
      reinterpret_cast<float *>(boxes_sorted.data_ptr()),
      reinterpret_cast<float *>(scores.data_ptr()),
      reinterpret_cast<int *>(keep.data_ptr()),
      num_boxes, iou_threshold);
  auto keep_cpu = keep.to(torch::kCPU);

  std::vector<int> keep_indices;
  auto keep_accessor = keep_cpu.accessor<int, 1>();
  for (int i = 0; i < num_boxes; ++i) {
    if (keep_accessor[i] == 1) {
      keep_indices.push_back(i);
    }
  }
  return torch::tensor(keep_indices, torch::TensorOptions().dtype(torch::kInt32));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(nms)
}
