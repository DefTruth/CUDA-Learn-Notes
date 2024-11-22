#include <vector>
#include <string>

struct Box {
  float x1, y1, x2, y2, score;
  float area() const {return (std::abs(x2 - x1 + 1)) * std::abs(y2 - y1 + 1); }
  float iou_of(const Box& other) const{
    float inner_x1 = x1 > other.x1 ? x1 : other.x1;
    float inner_y1 = y1 > other.y1 ? y1 : other.y1;
    float inner_x2 = x2 < other.x2 ? x2 : other.x2;
    float inner_y2 = y2 < other.y2 ? y2 : other.y2;
    float inner_h = inner_y2 - inner_y1 + 1.0f;
    float inner_w = inner_x2 - inner_x1 + 1.0f;
    float inner_area = inner_h * inner_w;
    return (inner_area / (area() + tbox.area() - inner_area));
  }
}

void hard_nms(std::vector<Box> &input, std::vector<Box> &output, 
              float iou_threshold){
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),[](Box& a, Box& b) { return a.score > b.score; });
  int box_num = input.size();
  std::vector<int> merged(box_num, 0);
  for (int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    merged[i] = 1;
    for (int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;
      float iou = input[i].iou_of(input[j]);
      if (iou > iou_threshold) merged[j] = 1;
    }
    output.push_back(input[i]);
  }
}