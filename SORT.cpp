#include "SORT.h"
#include "Hungarian.h"
#include "KalmanTracker.h"
#include <opencv2/opencv.hpp>

#include <set>
#include <vector>
#include <chrono>


double SORT::GetIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt){
    float in = (bb_dr & bb_gt).area();
    float un = bb_dr.area() + bb_gt.area() - in;

    if(un < DBL_EPSILON)
        return 0;

    double iou = in / un;

    return iou;
}


std::vector<SORT::TrackingBox> SORT::Sortx(std::vector<std::vector<float>> bbox, int fi){
    int max_age = 90;//max time object disappear
    int min_hits = 3; //min time target appear
    double iouThreshold = 0.3;//matching IOU

    // variables used in the sort-loop
    std::vector<SORT::TrackingBox> detData;
    std::vector<cv::Rect_<float>> predictedBoxes;
    std::vector<std::vector<double>> iouMatrix;
    std::vector<int> assignment;

    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    // result
    std::vector<cv::Point> matchedPairs;
    std::vector<SORT::TrackingBox> frameTrackingResult;

    unsigned int trkNum = 0;
    unsigned int detNum = 0;
    // time
    auto start_time = std::chrono::high_resolution_clock::now();

    // bounding boxes in a frame store in detFrameData
    for (int i = 0; i < bbox.size() ; i++){
        SORT::TrackingBox tb;
        tb.frame = fi + 1;
        tb.box = Rect_<float>(cv::Point_<float>(bbox[i][0], bbox[i][1]), cv::Point_<float>(bbox[i][2], bbox[i][3]));
        detData.push_back(tb);
    }

    detFrameData.push_back(detData);

    // std::cout << "reading bbox from yolo \n";
    // initialize kalman trackers using first detections.
    if(trackers.size() == 0){
        std::vector<SORT::TrackingBox> first_frame;

        for (unsigned int i = 0; i < detFrameData[fi].size(); i++){

            KalmanTracker trk = KalmanTracker(detFrameData[fi][i].box);

            trackers.push_back(trk);

        }
        // output the first frame detections
        for (unsigned int id = 0; id < detFrameData[fi].size(); id++){
            SORT::TrackingBox tb = detFrameData[fi][id];
            tb.id = id;
            first_frame.push_back(tb);
            //std::cout << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height  << std::endl;
        }
        return first_frame;
    }

    /*
    3.1. get predicted locations from existing trackers
    */
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        cv::Rect_<float> pBox = (*it).predict();
        //std::cout << pBox.x << " " << pBox.y << std::endl;
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

    /*
    3.2. associate detections to tracked object (both represented as bounding boxes)
    */
    trkNum = predictedBoxes.size();
    detNum = detFrameData[fi].size();
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));


    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    HungAlgo.Solve(iouMatrix, assignment);

    // find matches, unmatched_detections and unmatched_predictions
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

    /*
    3.3. updating trackers
    update matched trackers with assigned detections.
    each prediction is corresponding to a tracker
    */
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].update(detFrameData[fi][detIdx].box);
    }

    // create and initialize new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        KalmanTracker tracker = KalmanTracker(detFrameData[fi][umd].box);
        trackers.push_back(tracker);
    }

    // get trackers' output
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits || fi <= min_hits))
        {
            SORT::TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.frame = fi;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;

        // remove dead tracklet
        if (it != trackers.end() && (*it).m_time_since_update > max_age)
            it = trackers.erase(it);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    //std::cout << "SORT time : " << duration.count() << " ms" << std::endl;

    return frameTrackingResult;

}
