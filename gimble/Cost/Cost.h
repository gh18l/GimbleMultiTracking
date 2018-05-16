#ifndef __COST_HPP__
#define __COST_HPP__

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <unordered_map>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include "RobustStitcher/FeatureMatch.h"
#include "GenCameraDriver.h"

#include "matching/tracker.h"
#include "feature/FeatureTensor.h"


class Cost
{
public:
	Cost();
	~Cost();
private:
	std::shared_ptr<ModelDetection> detector = std::make_shared<ModelDetection>();
	std::shared_ptr<ModelDetection> detector_people = std::make_shared<ModelDetection>();
	std::shared_ptr<ModelDetection> detector_face = std::make_shared<ModelDetection>();
	std::vector<cv::Rect> img_rect;
	///////the number of blocks
	int rows_b = 1;
	int cols_b = 2;
	///////the number of input net
	int set_b = 2;

	///////set the face blocks
	double width_f = 450.0;
	double height_f = 400.0;

	//////when the tracking happens, up/downsampling the image to reduce runtime
	float sam_scale = 1.0f;
	cv::Rect crop_roi;

	/////isfind
	int isfind_time = 0;
	std::vector<cv::Point> isfind_vec;


public:
	cv::Point people_point;
	float people_constant = 1;
	float people_flow_gain = 1;

	//////video flow////
	cv::Ptr<cv::BackgroundSubtractor> bg_model = cv::createBackgroundSubtractorMOG2().dynamicCast<cv::BackgroundSubtractor>();
	bool smoothMask = 0;   
	bool update_bg_model = true;
	///cost///
	cv::Mat fre;
	int fre_tre = 10;

	////tracking
	bool istracking = 0;
	////tracking block
	float scale;
	float finalwidth, finalheight;

	////show
	cv::Mat show_opencv;
	int thread_flag = 0;

	int isfind_max;
	bool find_face = 0;

	int Thread_end = 0;
private:
	bool isfind();
public:
	int init_people_detection(cv::Mat img);
	int init_face_detection();
	bool iscontain(cv::Rect roi);
	int people_detection(cv::Mat& img);
	cv::Mat cal_dis(cv::Mat flow_uv0, cv::Mat flow_uv1);
	int flow(cv::Mat src1, cv::Mat src2, cv::Point &point);
	int video_updateflow(cv::Mat src, cv::Point& point);
	int video_initflow(cv::Mat src);
	int video_preflow(cv::Mat src);
	int SetBlock(cv::Mat img);
	cv::Mat SetFaceBlock(cv::Mat local, cv::Mat ref, cv::Point current_point);
	int GetSum(cv::Mat input);
	std::vector<int> choose_maxs(std::vector<int> sum);
	int find_min(std::vector<int> index, std::vector<int>sum);
	int video_updatedetection(cv::Mat src, cv::Point src_point, cv::Point& dst_point);

	int tracking_init(cv::Mat frame);
	int Thtracking();
	void startTh();

	double people_match(cv::Mat img1, cv::Mat img2);
	int face_detection(cv::Mat local, cv::Mat ref, cv::Point current_point);



private:
	
public:
	cv::Mat ref_people;
	std::vector<cv::Mat> current_show;
	int max_cap = 10000;
	std::vector<int>tracked_id;   //id has tracked
	std::unordered_map<int, int> trackid_showid;
	std::unordered_map<int, cv::Mat>superpixel_people;
	std::vector<cv::Mat>NeedToShow;
	int NeedToShow_index = 0;
	cv::Rect tracking_roi;
	int current_id;
	std::vector<int> is_face_id;
	void tracking(cv::Mat img);
	cv::Mat merge_img(std::vector<cv::Mat>frame_seg);
	int SeekNextDst(cv::Mat src, cv::Point& dst_point);
	void add_id(int current_id, cv::Mat img);

	static bool dotinrect(int x, int y, cv::Rect rect);
	//void onMouse(int event, int x, int y, int, void*);
	std::unordered_map<int, cv::Rect> current_tracking;
	bool finish_push; //finish push into current_tracking
	int mouse_index = 0;
	tracker mytracker;
};

#endif
