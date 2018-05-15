// include std
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include<fstream>
#include <queue>
#include <thread>
#include <memory>
#include <time.h>
#include "Serial.h"  
#include <string.h> 
//#include<tchar.h>
// opencv
#include "Display.h"
//
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"




// cuda
#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "GenCameraDriver.h"

#include "GimUtil.h"
#include "Cost.h"

#include "FeatureMatch.h"
#include "CameraParamEstimator.h"
#include "Compositor.h"
//imshow
//#define _SHOW

//#define DEBUG_
//#define SAVEAVI

using namespace std;

#define coe_gain 2.5
extern int showSub;

CSerial serial;
GimUtil gimutil;
Cost cost;
Display display;

calib::FeatureMatch match;
calib::CameraParamEstimator estimator;
calib::BundleAdjustment bundleAdjust;
calib::Compositor compositor;


std::vector<cam::GenCamInfo> camInfos;
std::shared_ptr<cam::GenCamera> cameraPtr;
int num=0;
cv::VideoWriter writer1, writer2;
bool start_flag = 0;
bool startTracking_flag = 0;

int delay_gimble()
{
	gimutil.sleep(1000);
	return 0;
}
int collectdelay_gimble()
{
	gimutil.sleep(2500);
	return 0;
}

void shoot(cv::Mat &ref, cv::Mat &local)
{
	char name[200], temp[50];  //
	cv::Mat local_bayer, ref_bayer;
	cv::Mat watching;
	std::vector<cam::Imagedata> imgdatas(2);
	cameraPtr->captureFrame(imgdatas);
	cv::Mat(camInfos[0].height, camInfos[0].width,
		CV_8U, reinterpret_cast<void*>(imgdatas[0].data)).copyTo(local_bayer);
	cv::Mat(camInfos[1].height, camInfos[1].width,
		CV_8U, reinterpret_cast<void*>(imgdatas[1].data)).copyTo(ref_bayer);

	//////////////convert/////////////
	cv::cvtColor(local_bayer, local, CV_BayerRG2BGR);
	cv::cvtColor(ref_bayer, ref, CV_BayerRG2BGR);
	std::vector<cv::Mat> channels(3);
	split(local, channels);
	channels[0] = channels[0] * camInfos[0].blueGain;
	channels[1] = channels[1] * camInfos[0].greenGain;
	channels[2] = channels[2] * camInfos[0].redGain;
	merge(channels, local);

	split(ref, channels);
	channels[0] = channels[0] * camInfos[1].blueGain;
	channels[1] = channels[1] * camInfos[1].greenGain;
	channels[2] = channels[2] * camInfos[1].redGain;
	merge(channels, ref);
#ifdef SAVEAVI
	writer1 << ref;
	writer2 << local;
#endif
#ifdef _SHOW
	cv::Mat show1, show2;
	cv::resize(local, show1, cv::Size(800, 600));
	cv::resize(ref, show2, cv::Size(800, 600));
	cv::imshow("local", show1);
	cv::imshow("ref", show2);
	cv::waitKey(30);
#endif
}

void gimbal_init(float delta_Yaw, float delta_Pitch)
{
	gimutil.current_pulse.x = delta_Yaw + YM;
	gimutil.current_pulse.y = delta_Pitch + PM;
	serial.Serial_Send_Yaw(YM + delta_Yaw);     //��������ľ�̬��Ա����
	serial.Serial_Send_Pitch(PM + delta_Pitch);
	collectdelay_gimble();
}

void init(cv::Point init)    //�����˹����
{
	// init camera
	cameraPtr = cam::createCamera(cam::CameraModel::XIMEA_xiC);
	cameraPtr->init();
	// set camera setting
	cameraPtr->startCapture();
	cameraPtr->setFPS(-1, 20);
	cameraPtr->setAutoExposure(-1, cam::Status::on);
	cameraPtr->setAutoExposureLevel(-1, 40);
	cameraPtr->setAutoWhiteBalance(-1);
	cameraPtr->makeSetEffective();
	// set capturing setting
	cameraPtr->setCamBufferType(cam::GenCamBufferType::Raw);
	cameraPtr->setCaptureMode(cam::GenCamCaptureMode::Continous, 40);
	cameraPtr->setCapturePurpose(cam::GenCamCapturePurpose::Streaming);
	//cameraPtr->setVerbose(true);
	// get camera info
	cameraPtr->getCamInfos(camInfos);
	cameraPtr->startCaptureThreads();
	cameraPtr->makeSetEffective();

	//init serial
	serial.Serial_Init();

	//init gimble
	gimutil.init_stitcher();
	cv::Mat ref, local;
	gimbal_init(init.x, init.y);    //ת��0,0���崦
	shoot(ref, local);
	gimutil.find_position(ref, local, gimutil.current_point);    //update now point
	std::cout << gimutil.current_point << std::endl;
	//init picture parameters
}
void camera_close()
{
	cameraPtr->stopCaptureThreads();
	cameraPtr->release();
}

void onMouse(int event, int x, int y, int, void*)
{
	if(event != cv::EVENT_LBUTTONDOWN)
		return ;
	//std::cout<<"pause                       ......................"<<std::endl;
	if(cost.finish_push == 0)
	{
		while(1)
		{
			if(cost.finish_push == 1)
				break;
		}
	}
	std::cout<<"pause                       ......................"<<std::endl;
	for(std::unordered_map<int, cv::Rect>::iterator iter = cost.current_tracking.begin(); iter != cost.current_tracking.end(); iter++)
	{
		if(Cost::dotinrect(x, y, iter->second) == 1)
		{
			std::unordered_map<int, cv::Mat>::iterator super = cost.superpixel_people.find(iter->first);
			if(super == cost.superpixel_people.end())
				break;
			else
			{
				if(cost.NeedToShow.size() < showSub)
					cost.NeedToShow.push_back(cost.superpixel_people[iter->first]);
				else
				{
					cost.superpixel_people[iter->first].copyTo(cost.NeedToShow[cost.mouse_index]);
					cost.mouse_index++;
					if(cost.mouse_index > 5)
						cost.mouse_index = 0;
				}	
			}
			
		}
	}
}




void collection(std::vector<cv::Mat>& imgs)
{
	std::string datapath = "E:/data/bb";
	cv::Mat ref, local;
	gimutil.current_pulse.x = YM+180;
	gimutil.current_pulse.y = PM+120;
	serial.Serial_Send_Yaw(gimutil.current_pulse.x);
	serial.Serial_Send_Pitch(gimutil.current_pulse.y);
	collectdelay_gimble();

	for (int i = 0; i < gimutil.Row; i++)
	{
		if (i % 2 == 0)
		{
			for (int j = 0; j < gimutil.Col; j++)
			{
				gimutil.current_pulse.x = YM + 180 - gimutil.dX * j;
				gimutil.current_pulse.y = PM + 120 - gimutil.dY * i;
				serial.Serial_Send_Yaw(gimutil.current_pulse.x);
				serial.Serial_Send_Pitch(gimutil.current_pulse.y);
				collectdelay_gimble();
				for (int k = 0; k < 1; k++)
				{
					shoot(ref, local);
					local.copyTo(imgs[j*gimutil.Row + i]);
					cv::imwrite(cv::format("%s/local_%d_%d.png",
						datapath.c_str(), int(gimutil.dX) * j, int(gimutil.dY) * i), local);
				}
			}
		}
		
		if (i % 2 == 1)
		{
			for (int j = gimutil.Col-1; j >= 0; j--)
			{
				gimutil.current_pulse.x = YM + 180 - gimutil.dX * j;
				gimutil.current_pulse.y = PM + 120 - gimutil.dY * i;
				serial.Serial_Send_Yaw(gimutil.current_pulse.x);
				serial.Serial_Send_Pitch(gimutil.current_pulse.y);
				collectdelay_gimble();
				for (int k = 0; k < 1; k++)
				{
					shoot(ref, local);
					local.copyTo(imgs[j*gimutil.Row + i]);
					cv::imwrite(cv::format("%s/local_%d_%d.png",
						datapath.c_str(), int(gimutil.dX) * j, int(gimutil.dY) * i), local);
				}
			}
		}
	}
}


bool isOverlap(const cv::Rect &rc1, const cv::Rect &rc2)
{
	if (rc1.x + rc1.width  > rc2.x &&
		rc2.x + rc2.width  > rc1.x &&
		rc1.y + rc1.height > rc2.y &&
		rc2.y + rc2.height > rc1.y
		)
		return true;
	else
		return false;
}
int findoverlap(cv::Point corner_current, cv::Size size_current, vector<cv::Point>& corners, vector<cv::Size>& sizes, std::vector<int>& index)
{
	cv::Rect Rect_current(corner_current, size_current);
	for (int i = 0; i < gimutil.Row*gimutil.Col; i++)
	{
		cv::Rect temp(corners[i], sizes[i]);
		if (isOverlap(Rect_current, temp))
		{
			index.push_back(i);   //���������ͼ�����ţ��ù�����������
		}
	}
	return 0;
}
int save_para(vector<calib::CameraParams>& cameras, vector<cv::Point>& corners, vector<cv::Size>& sizes)
{
	ofstream para;
	cv::Mat K;
	para.open("/home/lgh/Documents/gimbleMulti/para.txt", ios::out);
	if (!para)
		cout << "No have txt" << endl;
	for (int i = 0; i < cameras.size(); i++)
	{
		para << cameras[i].focal << " " << cameras[i].aspect << " "
			<< cameras[i].ppx << " " << cameras[i].ppy << " ";
		//���Կ��ǿ���Mat_ģ���࣬K()const������������camera.cpp��
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				para << cameras[i].R.at<double>(j, k) << " ";
			}
		}
		para << corners[i].x << " " << corners[i].y << " " << sizes[i].width << " " << sizes[i].height << " ";
		/*for (int j = 0; j < 3; j++)
		{
		para << cameras[i].t.at<float>(j,0) << " ";
		}*/
		para << endl;
	}
	para.close();

	return 0;
}

int read_para(vector<calib::CameraParams> &cameras, vector<cv::Point> &corners, vector<cv::Size>&sizes)
{
	ifstream para;
	cv::Mat K;
	para.open("/home/lgh/Documents/gimbleMulti/para.txt");
	if (!para.is_open())
	{
		cout << "can not open txt" << endl;
		return -1;
	}
	string str;

	for (int i = 0; i < gimutil.Row*gimutil.Col; i++)   //����û���Զ�����ͼƬ����������������
	{
		getline(para, str, ' ');
		cameras[i].focal = stof(str);
		getline(para, str, ' ');
		cameras[i].aspect = stof(str);
		getline(para, str, ' ');
		cameras[i].ppx = stof(str);
		getline(para, str, ' ');
		cameras[i].ppy = stof(str);
		cameras[i].R.create(3, 3, CV_64FC1);
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				getline(para, str, ' ');
				cameras[i].R.at<double>(j, k) = stof(str);
			}
		}
		getline(para, str, ' ');
		corners[i].x = stoi(str);
		getline(para, str, ' ');
		corners[i].y = stoi(str);
		getline(para, str, ' ');
		sizes[i].width = stoi(str);
		getline(para, str, ' ');
		sizes[i].height = stoi(str);
		/*for (int j = 0; j < 3; j++)
		{
		getline(para, str, ' ');
		cameras[i].t.at<float>(j, 0) = stof(str);
		}*/
		getline(para, str);
	}
	para.close();
	return 0;
}

int GetCurrentPara(vector<calib::CameraParams>& cameras, cv::Point2f current_point, calib::CameraParams &current_para)
{
	///build a para matrix
	cv::Mat focals(gimutil.Row, gimutil.Col, CV_64FC1), ppxs(gimutil.Row, gimutil.Col, CV_64FC1), ppys(gimutil.Row, gimutil.Col, CV_64FC1);
	int index = 0;
	//////R and T
	vector<cv::Mat>Rs(9);   //ÿ���Ӧһ��rԪ��, ÿ��Ԫ�صĸ�����ӦͼƬ����
	for (int i = 0; i < 9; i++)
	{
		Rs[i].create(gimutil.Row, gimutil.Col, CV_64FC1);
	}
	//vector<cv::Mat>Ts(3);
	for (int i = 0; i < gimutil.Col; i++)      //�������¶�����
	{
		for (int j = 0; j < gimutil.Row; j++)
		{
			focals.at<double>(j, i) = cameras[index].focal;      //��j�е�i��
			ppxs.at<double>(j, i) = cameras[index].ppx;
			ppys.at<double>(j, i) = cameras[index].ppy;
			////ÿ����һ��klѭ���������һ��λ�ñ����� ���Ϊ9
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					Rs[k * 3 + l].at<double>(j, i) = cameras[index].R.at<double>(k, l); //
				}
			}
			index++;
		}
	}

	///////////////����������Ҫ�ӵ�readpara�������
	vector<double>value;
	vector<float>mapx(1, (current_point.x - gimutil.X_MIN)*1.0 / gimutil.dX);    //Ĭ�������00��ʼ
	vector<float>mapy(1, (current_point.y - gimutil.Y_MIN)*1.0 / gimutil.dY);    //Ĭ�������00��ʼ
	cv::remap(focals, value, mapx, mapy, cv::INTER_LINEAR);    //������Ҫclearһ�£���ʵ�鷵��ֵ��push���Ǹ���
	current_para.focal = value[0];
	value.clear();

	cv::remap(ppxs, value, mapx, mapy, cv::INTER_LINEAR);
	current_para.ppx = value[0];
	value.clear();

	cv::remap(ppys, value, mapx, mapy, cv::INTER_LINEAR);
	current_para.ppy = value[0];
	value.clear();
	current_para.R.create(3, 3, CV_64FC1);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cv::remap(Rs[i * 3 + j], value, mapx, mapy, cv::INTER_LINEAR);
			current_para.R.at<double>(i, j) = value[0];
			value.clear();
		}
	}
	return 0;
}
int panoram(cv::Mat& result,std::vector<cv::Mat>& imgs)
{
	collection(imgs);
	int img_num = imgs.size();
	std::cout << img_num << std::endl;
	cv::Mat connection;
	connection = cv::Mat::zeros(img_num, img_num, CV_8U);
	for (size_t i = 0; i < imgs.size(); i++) {
		for (size_t j = 0; j < imgs.size(); j++) {
			if (i == j)
				continue;
			int row1 = i / gimutil.Row;
			int col1 = i % gimutil.Row;
			int row2 = j / gimutil.Row;
			int col2 = j % gimutil.Row;
			if (abs(row1 - row2) <= 1 && abs(col1 - col2) <= 1) {
				connection.at<uchar>(i, j) = 1;
				connection.at<uchar>(j, i) = 1;
			}
		}
	}

	match.init(imgs, connection);
	match.match();
	//match.debug();

	calib::CameraParamEstimator estimator;
	estimator.init(imgs, connection, match.getImageFeatures(), match.getMatchesInfo());
	estimator.estimate();
	
	calib::Compositor compositor;
	compositor.init(imgs, estimator.getCameraParams());
	compositor.composite();
	save_para(compositor.cameras, compositor.corners, compositor.sizes);

	return 0;
}
int warp(std::vector<cv::Mat>& imgs, vector<calib::CameraParams> &cameras, 
			vector<cv::Point> &corners, vector<cv::Size>&sizes, cv::Mat& src, cv::Point2f current_pulse, 
				calib::CameraParams& current_para, std::vector<calib::Imagefeature>& features)
{
	cv::Ptr<cv::detail::SphericalWarper> w = cv::makePtr<cv::detail::SphericalWarper>(false);
	std::shared_ptr<cv::detail::Blender> blender_ = std::make_shared<cv::detail::MultiBandBlender>(false);
	clock_t start, finish;
	current_pulse.x = YM + 180 - current_pulse.x;
	current_pulse.y = PM + 120 - current_pulse.y;
	GetCurrentPara(cameras, current_pulse, current_para);
	/////////���corner_current��size_current///////////////
	cv::Mat src_warped, mask, mask_warped;
	cv::Point corner_current;
	cv::Size size_current;
	w->setScale(16000);
	// calculate warping filed
	cv::Mat K, R;
	current_para.K().convertTo(K, CV_32F);
	current_para.R.convertTo(R, CV_32F);
	cv::Mat initmask(src.rows, src.cols, CV_8U);
	initmask.setTo(cv::Scalar::all(255));
	corner_current = w->warp(src, K, R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, src_warped);
	w->warp(initmask, K, R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);
	size_current = mask_warped.size();

	//calib::Compositor compositor;
	std::vector<int> index;
	findoverlap(corner_current, size_current, corners, sizes, index);   //���compositor.index

	calib::FeatureMatch match;
	calib::Imagefeature current_feature;
	std::vector<calib::Matchesinfo> current_matchesInfo(index.size());
	for (int i = 0; i < index.size(); i++)
	{
		match.current_feature_thread_(src, features[index[i]], current_feature, current_matchesInfo[i], i);    //������ƴ��ͼ�����Χͼ�������ƥ��
	}
	
	//calib::BundleAdjustment bundleAdjust;
	calib::BundleAdjustment bundleAdjust;
	bundleAdjust.refine_BA(index, current_feature, features, current_matchesInfo, cameras, current_para);    
	
// #ifdef DEBUG_
// 	for (int i = 0; i < index.size(); i++)
// 	{
// 		/*if (current_matchesInfo[i].confidence < 1.5)
// 		continue;*/
// 		if (current_matchesInfo[i].confidence < 2.9)
// 			continue;
// 		cv::Mat result;
// 		// make result image
// 		int width = src.cols * 2;
// 		int height = src.rows;
// 		result.create(height, width, CV_8UC3);
// 		cv::Rect rect(0, 0, src.cols, height);
// 		src.copyTo(result(rect));
// 		rect.x += src.cols;
// 		imgs[index[i]].copyTo(result(rect));
// 		// draw matching points
// 		cv::RNG rng(12345);
// 		int r = 3;
// 		for (size_t kind = 0; kind < current_matchesInfo[i].matches.size(); kind++) {
// 			if (current_matchesInfo[i].inliers_mask[kind]) {
// 				cv::Scalar color = cv::Scalar(rng.uniform(0, 255),
// 					rng.uniform(0, 255), rng.uniform(0, 255));
// 				const cv::DMatch& m = current_matchesInfo[i].matches[kind];
// 				cv::Point2f p1 = current_feature.keypt[m.queryIdx].pt;
// 				cv::Point2f p2 = features[i].keypt[m.trainIdx].pt;
// 				p2.x += src.cols;
// 				cv::circle(result, p1, r, color, -1, cv::LINE_8, 0);
// 				cv::circle(result, p2, r, color, -1, cv::LINE_8, 0);
// 				//cv::line(result, p1, p2, color, 5, cv::LINE_8, 0);
// 			}
// 		}
// 		cv::imwrite(cv::format("E:/code/project/gimble3.23/features/matching_points_%02d_%02d.jpg", -1, index[i]),
// 			result);
// 	}
// #endif

	return 0;
}

int main(int argc, char* argv[]) {
	clock_t start, finish;

	cv::Mat result;
	std::vector<cv::Mat> imgs;
	vector<calib::CameraParams> cameras(gimutil.Row*gimutil.Col);
	vector<cv::Point> corners(gimutil.Row*gimutil.Col);
	vector<cv::Size> sizes(gimutil.Row*gimutil.Col);
	cv::Mat ref, local;
	init(cv::Point(0, 0));
	// while(1)
	// {
		
	// 	shoot(ref,local);
	// 	cv::resize(ref,ref,cv::Size(800,600));
	// 	cv::resize(local,local,cv::Size(800,600));
	// 	cv::imshow("1",ref);
	// 	cv::imshow("2",local);
	// 	cv::waitKey(1);
	// }
	if (argc > 1)//�в������п��ܲ���Ҫɨ����
	{
		std::string panoname = std::string(argv[1]);
		result = cv::imread("/home/lgh/Documents/gimbleMulti/1.png");
		cv::resize(result, result, cv::Size(4511,3930));
		std::string datapath = "/home/lgh/Documents/gimbleMulti/bb";
		for (int i = 0; i <= 280; i = i + 40) {
		for (int j = 0; j <= 240; j = j + 30) {
			imgs.push_back(cv::imread(cv::format("%s/local_%d_%d.png", 
				datapath.c_str(), i, j)));
			}
		}
	}
	else
	{
		imgs.resize(gimutil.Row*gimutil.Col);
		panoram(result, imgs);
	}
	//////////read parameters///////////
	read_para(cameras, corners, sizes);
	std::vector<calib::Imagefeature> features;
	match.read_features(features);
	//global-result warping
	cv::Mat result_temp;

	cv::Point dst_point,src_point;
	serial.Serial_Send_Yaw(YM);
	serial.Serial_Send_Pitch(PM);
	gimutil.current_pulse.x = YM;
	gimutil.current_pulse.y = PM;
	collectdelay_gimble();
	shoot(ref, local);

	gimutil.find_position(ref, local, gimutil.current_point);
	src_point = gimutil.current_point;

	corners.resize(cameras.size() + 1);
	sizes.resize(cameras.size() + 1);
	for (int i = 0; i < 100; i++)
	{
		cost.video_preflow(ref);
		shoot(ref, local);
		std::cout << i << std::endl;
	}
	calib::CameraParams current_para;
	calib::Compositor compositor;
	display.display_init(result);
	cost.init_people_detection(ref);
	cost.init_face_detection();

	cv::namedWindow("tracking");
	cv::setMouseCallback("tracking", onMouse, 0);
	cv::Mat ref_temp;
	int index = 0;
	cost.startTh();
	while (1)
	{
		result.copyTo(result_temp);
		
		shoot(ref, local);
		ref.copyTo(ref_temp);
		gimutil.colorCorrectRGB(local, result_temp);
		//set current_id and dst_point, decides to see 
		if (index > 20 && cost.SeekNextDst(ref_temp, dst_point) == 0 ) 
		{
			if(gimutil.detect_move(dst_point) != 0)
			{
				continue;
			}
			gimutil.move(dst_point);
			serial.Serial_Send_Yaw(gimutil.current_pulse.x);
			serial.Serial_Send_Pitch(gimutil.current_pulse.y);
			delay_gimble();
			shoot(ref, local);
			gimutil.gimble_find_position(ref, local,
					gimutil.current_point, 2, gimutil.current_point);
			ref.copyTo(ref_temp);
			gimutil.colorCorrectRGB(local, result_temp);
			cost.face_detection(local, ref_temp, gimutil.current_point);
			warp(imgs, cameras, corners, sizes, local, gimutil.current_pulse,
					current_para, features); 
			
			index = 0;
		}
		if (start_flag == 0)
		{
			warp(imgs, cameras, corners, sizes, local, gimutil.current_pulse,
				current_para, features);
			start_flag = 1;
		}
		compositor.single_composite(current_para, local, result_temp, corners, sizes);
		resize(result_temp, result_temp, cv::Size((result_temp.cols / 100) * 100, (result_temp.rows / 100) * 100));
		std::cout<<"need to show id      "<<cost.NeedToShow.size()<<std::endl;


		////////////////////////
		if (cost.NeedToShow.size() == 0)
		{
			cv::Mat black = cv::Mat::zeros(600, 600, CV_8UC3);
			std::vector<cv::Mat> blackvec;
			blackvec.push_back(black);
			display.display(result_temp, blackvec);
		}
		else
		{
			display.display(result_temp, cost.NeedToShow);
		}
		if (cost.thread_flag == 1) 
		{
			cv::Mat tracking_temp;
			cv::resize(cost.show_opencv, tracking_temp, cv::Size(1200, 800));
			cv::imshow("tracking", tracking_temp);
			cv::waitKey(30);
		}

		index++;
	}
	camera_close();   //////////////////////
	return 0;
}
