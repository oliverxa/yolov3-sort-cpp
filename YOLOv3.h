#ifndef YOLOV3_H
#define YOLOV3_H
#include "Darknet.h"
#include "opencv2/opencv.hpp"

#include <torch/torch.h>
#include <vector>

class YOLOv3{

public:

    std::vector<std::vector<float>> yolov3(Darknet net, cv::Mat origin_image, torch::Device device);
};

#endif // YOLOV3_H
