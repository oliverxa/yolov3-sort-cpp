#ifndef SORT_H
#define SORT_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "KalmanTracker.h"

//using namespace std;

class SORT
{
private:

public:
    typedef struct TrackingBox{
        int frame;
        int id;
        cv::Rect_<float> box;
    }TrackingBox;

    std::vector<KalmanTracker> trackers;
    std::vector<std::vector<TrackingBox>> detFrameData;

    std::vector<TrackingBox> Sortx(std::vector<std::vector<float>> bbox, int fi);
    double GetIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt);

};


#endif


