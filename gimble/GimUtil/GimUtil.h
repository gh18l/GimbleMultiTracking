#ifndef __PROPERTY_HPP__
#define __PROPERTY_HPP__

#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include <thread>
#ifdef WIN32
#include <Windows.h>
#endif
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc.hpp>

#define PM 6200
#define YM 3900

class GimUtil{
public:
    static int sleep(int miliseconds);
    cv::Mat SEDDetector(cv::Mat img, float scale);
    int colorCorrectRGB(cv::Mat & srcImg, cv::Mat dstImg);
    bool isInside(cv::Point2f pt, cv::Rect rect);
public:
    GimUtil();
    ~GimUtil();
    int move(cv::Point& dst);
    int detect_move(cv::Point dst);
    int find_position(cv::Mat refImg, cv::Mat localImg, cv::Point &out_point);
	int init_stitcher();
	int gimble_find_position(cv::Mat refImg, cv::Mat localImg, cv::Point ref_point, float region_mul,cv::Point &out_point);
    void get_move_coe();
public:
    float dX = 40.0, X_MIN = 0, dY = 30.0, Y_MIN = 0;
	int Row = 11;
	int Col = 9;
    cv::Point current_point;
	cv::Point2f current_pulse;
    cv::Size sizeBlock;
    float move_coe_x = 6.4;   //4.0*1.6 -> 25mm    4.0 -> 16mm
    float move_coe_y = 4.8;   //3.0*1.6 -> 25mm    3.0 -> 16mm
public:
    cv::Ptr<cv::ximgproc::StructuredEdgeDetection> ptr;
    float scale;
    int video_width;
    int video_height;
};
#endif
