#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Darknet.h"

#include <unistd.h>
#include <set>
#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"



std::vector<vector<float>> yolov3(Darknet net, cv::Mat image, torch::Device device);

typedef struct TrackingBox{
    int frame;
    int id;
    cv::Rect_<float> box;
}TrackingBox;

double GetIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt){
    float in = (bb_dr & bb_gt).area();
    float un = bb_dr.area() + bb_gt.area() - in;

    if(un < DBL_EPSILON)
        return 0;

    double iou = in / un;

    return iou;
}

// global variables for counting
#define CNUM 20
int total_frames = 0;
double total_time = 0.0;

std::vector<TrackingBox> SORT(std::vector<vector<float>> bbox);

int main(int argc, const char* argv[])
{
    std::cerr << "usage: yolo-app <image path>\n";

    // yolov3 init
    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
        std::cout << "GPU--version"<< endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "CPU--version"<< endl;
    }

    torch::Device device(device_type);

    // input image size for YOLO v3

    Darknet net("../models/yolov3-swim-test.cfg", &device);

    int input_image_size = 416;

    std::map<string, string> *info = net.get_net_info();
    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    net.load_weights("../models/yolov3-swim_final.weights");
    std::cout << "weight loaded ..." << endl;

    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "loading video ..." << endl;
    cv::VideoCapture capture(argv[1]);

    if(!capture.isOpened()){
        std::cout << "load video Failed" << std::endl;
        return -1;
    }

    int frame_num = capture.get(cv::CAP_PROP_FRAME_COUNT);
    cv::Mat origin_image;

    std::cout << "start to inference ..." << endl;
    for(int i = 0; i < frame_num; i++){
        capture >> origin_image;
        std::vector<vector<float>> bbox = yolov3(net, origin_image, device);

        for(int i = 0; i < bbox.size(); i++){
            cv::rectangle(origin_image, cv::Point(bbox[i][0], bbox[i][1]), cv::Point(bbox[i][2], bbox[i][3]), cv::Scalar(0, 0, 255), 1, 1, 0);
            std::cout << bbox[i][0] << ' ' << bbox[i][1] << ' ' << bbox[i][2] << ' ' << bbox[i][3] << '\n' ;

        }

        // using sort for tracking
        SORT(bbox);

        cv::Mat output_image;
        cv::resize(origin_image, output_image, cv::Size(600, 600));
        cv::imshow("video test", output_image);
        if( cv::waitKey(10) == 27 ) break;
    }

    //cv::destroyWindow("video test");
	capture.release();
    std::cout << "Done" << endl;

    return 0;
}


std::vector<vector<float>> yolov3(Darknet net, cv::Mat origin_image, torch::Device device){

    // origin_image = cv::imread("../139.jpg");
    //origin_image = cv::imread(path);
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
    std::cout << "inference taken : " << duration.count() << " ms" << endl;

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
            tmp_box.push_back(result_data[i][1]);
            tmp_box.push_back(result_data[i][2]);
            tmp_box.push_back(result_data[i][3]);
            tmp_box.push_back(result_data[i][4]);
            Bbox.push_back(tmp_box);
        }

    }

    return Bbox;
}

std::vector<KalmanTracker> trackers;
int frame_count = 0;

std::vector<TrackingBox> SORT(std::vector<vector<float>> bbox){
    // read bounding box for matching

	int max_age = 1;
	int min_hits = 3; //min time target appear
	double iouThreshold = 0.3;
	std::vector<TrackingBox> detData;

	KalmanTracker::kf_count = 0;
	std::vector<cv::Rect_<float>> predictedBoxes;
	std::vector<std::vector<double>> iouMatrix;
	std::vector<int> assignment;

	std::set<int> unmatchedDetections;
	std::set<int> unmatchedTrajectories;
	std::set<int> allItems;
	std::set<int> matchedItems;

    // result
	std::vector<cv::Point> matchedPairs;
	std::vector<TrackingBox> frameTrackingResult;

	unsigned int trkNum = 0;
	unsigned int detNum = 0;
/*
	cv::RNG rng(0xFFFFFFFF);
	cv::Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
*/
    int start_time = cv::getTickCount();
    double cycle_time = 0.0;

    // bounding boxes in a frame store in detFrameData
    for (int i = 0; i < bbox.size() ; i++){
        TrackingBox tb;
        tb.frame = frame_count;
        tb.box = Rect_<float>(cv::Point_<float>(bbox[i][0], bbox[i][1]), cv::Point_<float>(bbox[i][2], bbox[i][3]));
        detData.push_back(tb);
    }

    std::vector<std::vector<TrackingBox>> detFrameData;
    detFrameData.push_back(detData);
    detData.clear();

    std::cout << "reading bbox from yolo \n";
    // initialize kalman trackers using first detections.
    if(trackers.size() == 0){
        std::vector<TrackingBox> first_frame;

        for (unsigned int i = 0; i < detFrameData[frame_count].size(); i++){

            KalmanTracker trk = KalmanTracker(detFrameData[frame_count][i].box);

            trackers.push_back(trk);

        }
        // output the first frame detections
        for (unsigned int id = 0; id < detFrameData[frame_count].size(); id++){
            TrackingBox tb = detFrameData[frame_count][id];
            tb.id = id;
            first_frame.push_back(tb);
            std::cout << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << std::endl;
        }
        return first_frame;
    }

    ///////////////////////////////////////
    // 3.1. get predicted locations from existing trackers.
    predictedBoxes.clear();

    for (auto it = trackers.begin(); it != trackers.end();)
    {
        cv::Rect_<float> pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            it++;
        }
        else
        {
            it = trackers.erase(it);
            //cerr << "Box invalid at frame: " << frame_count << endl;
        }
    }

    trkNum = predictedBoxes.size();
    detNum = detFrameData[frame_count].size();

    iouMatrix.clear();
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[frame_count][j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);

    // find matches, unmatched_detections and unmatched_predictions
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();

    if (detNum > trkNum) //	there are unmatched detections
    {
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        // calculate the difference between allItems and matchedItems, return to unmatchedDetections
        std::set_difference(allItems.begin(), allItems.end(),
            matchedItems.begin(), matchedItems.end(),
            insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }
    else
        ;

    // filter out matched with low IOU
    // output matchedPairs
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }


    ///////////////////////////////////////
    // 3.3. updating trackers

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x; //trkNum
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].update(detFrameData[frame_count][detIdx].box);
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        KalmanTracker tracker = KalmanTracker(detFrameData[frame_count][umd].box);
        trackers.push_back(tracker);
    }

    // get trackers' output
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
        {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.frame = frame_count;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;

        // remove dead tracklet
        if (it != trackers.end() && (*it).m_time_since_update > max_age)
            it = trackers.erase(it);
    }

    std::cout << "end" << std::endl;
    frame_count++;
    return frameTrackingResult;

}

