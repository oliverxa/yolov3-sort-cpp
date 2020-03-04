#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <chrono>
#include <string.h>

#include "Darknet.h"
#include "SORT.h"
#include "YOLOv3.h"


// global variables for counting
#define CNUM 20
//int total_frames = 0;
//double total_time = 0.0;

int main(int argc, const char* argv[])
{
    // yolov3 init
    torch::DeviceType device_type;
    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
        std::cout << "GPU--version"<< std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "CPU--version"<< std::endl;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    int input_image_size = 416;

    // std::string cfg, weight;
    // cfg = "../models/yolov3-swim-test.cfg";
    // weight = "../models/yolov3-swim_final.weights";

    std::cout << "loading configure file..." << std::endl;
    Darknet net("../models/yolov3-swim-test.cfg", &device);
    std::map<std::string, std::string> *info = net.get_net_info();
    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight model..." << std::endl;
    net.load_weights("../models/yolov3-swim_final.weights");
    //std::cout << "weight loaded ..." << std::endl;

    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "loading video ..." << std::endl;
    cv::VideoCapture capture(argv[1]);

    if(!capture.isOpened()){
        std::cout << "load video Failed" << std::endl;
        return -1;
    }

    int frame_num = capture.get(cv::CAP_PROP_FRAME_COUNT);
    cv::Mat origin_image;

    std::cout << "start to inference ..." << std::endl;

    SORT Sort;
    YOLOv3 Yolov3;

    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each video.

    cv::RNG rng(0xFFFFFFFF);
	cv::Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);

    for(int fi = 0; fi < frame_num; fi++){
        capture >> origin_image;

        std::vector<std::vector<float>> bbox = Yolov3.yolov3(net, origin_image, device);
        //std::cout << bbox.size();
/*
        for(int i = 0; i < bbox.size(); i++){
            cv::rectangle(origin_image, cv::Point(bbox[i][0], bbox[i][1]), cv::Point(bbox[i][2], bbox[i][3]), cv::Scalar(0, 0, 255), 1, 1, 0);
            std::cout << bbox[i][0] << ' ' << bbox[i][1] << ' ' << bbox[i][2] << ' ' << bbox[i][3] << '\n' ;

        }
*/
        // using sort for tracking
        //std::vector<Sort::TrackingBox> frameTrackingResult = Sort::Sort(bbox, fi);
        std::vector<SORT::TrackingBox> frameTrackingResult = Sort.Sortx(bbox, fi);
        //std::cout << frameTrackingResult.size();

        for(auto tb : frameTrackingResult){
            cv::rectangle(origin_image, tb.box, randColor[tb.id % CNUM], 6, 8, 0);
            cv::Point pt = cv::Point(tb.box.x, tb.box.y);
            cv::Scalar color = cv::Scalar(0, 255, 255);
            std::string num = std::to_string(tb.id);
            cv::putText(origin_image, num, pt, cv::FONT_HERSHEY_DUPLEX, 4.0, color, 2);
        }

        Size size(800,600);
        cv::resize(origin_image, origin_image, size);
        cv::imshow("video test", origin_image);
        if( cv::waitKey(10) == 27 ) break;
    }

    cv::destroyWindow("video test");
	capture.release();
    std::cout << "Done" << std::endl;

    return 0;
}
