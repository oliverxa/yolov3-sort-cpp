#include <vector>
#include <chrono>
#include "YOLOv3.h"
#include "Darknet.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>


std::vector<std::vector<float>> YOLOv3::yolov3(Darknet net, cv::Mat origin_image, torch::Device device){

    int input_image_size = 416;

    cv::Mat resized_image;
    cv::cvtColor(origin_image, resized_image,  cv::COLOR_BGR2RGB);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0/255);

    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
    img_tensor = img_tensor.permute({0,3,1,2});

    auto start = std::chrono::high_resolution_clock::now();

    auto output = net.forward(img_tensor);

    auto result = net.write_results(output, 80, 0.6, 0.4);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // It should be known that it takes longer time at first time
    std::cout << "Yolo time : " << duration.count() << " ms" << endl;

    std::vector<vector<float>> Bbox;

    if (result.dim() == 1)
    {
        std::cout << "no object found" << endl;
    }
    else
    {
        int obj_num = result.size(0);

        std::cout << obj_num << " objects found" << endl;

        float w_scale = float(origin_image.cols) / input_image_size;
        float h_scale = float(origin_image.rows) / input_image_size;

        result.select(1,1).mul_(w_scale);
        result.select(1,2).mul_(h_scale);
        result.select(1,3).mul_(w_scale);
        result.select(1,4).mul_(h_scale);

        auto result_data = result.accessor<float, 2>();

        // xmin, ymin, xmax, ymax
        for (int i = 0; i < result.size(0) ; i++)
        {
            std::vector<float> tmp_box;
            //cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
            //std::cout << result_data[i][1] << ' ' << result_data[i][2] << ' ' << result_data[i][3] << ' ' << result_data[i][4] << '\n' ;
            for(int j=1; j<=4; j++){
                tmp_box.push_back(result_data[i][j]);
            }
            Bbox.push_back(tmp_box);
        }

    }

    return Bbox;
}
