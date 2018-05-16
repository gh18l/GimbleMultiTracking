#include "Cost.h"
#include<opencv2/optflow.hpp>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <unordered_map>
#include "GenCameraDriver.h"
using namespace std;

extern std::vector<cam::GenCamInfo> camInfos;
extern std::shared_ptr<cam::GenCamera> cameraPtr;
#define _DEBUG_COST     //ע�͵�Ϊreleaseģʽ
//#define VISUALIZE
#define args_nn_budget 100  
#define args_max_cosine_distance 0.2
#define args_min_confidence 0.35  
#define args_nms_max_overlap 1.0

extern int showSub;

Cost::Cost() 
{
	current_show.resize(2);   //ֻ��ʾ����
}
Cost::~Cost() {}

bool Cost::dotinrect(int x, int y, cv::Rect rect)
{
	if(x>rect.x && x<rect.x+rect.width && y>rect.y && y<rect.y+rect.height)
		return 1;
	else
		return 0;
}

// void Cost::onMouse(int event, int x, int y, int, void*)
// {
// 	if(event != cv::EVENT_LBUTTONDOWN)
// 		return ;
// 	if(finish_push == 0)
// 	{
// 		while(1)
// 		{
// 			if(finish_push == 1)
// 				break;
// 		}
// 	}

// 	for(std::unordered_map<int, cv::Rect>::iterator iter = current_tracking.begin(); iter != current_tracking.end(); iter++)
// 	{
// 		if(dotinrect(x, y, iter->second) == 1)
// 		{
// 			std::unordered_map<int, cv::Mat>::iterator super = superpixel_people.find(iter->first);
// 			if(super == superpixel_people.end())
// 				break;
// 			else
// 			{
// 				if(NeedToShow.size() < showSub)
// 					NeedToShow.push_back(superpixel_people[iter->first]);
// 				else
// 				{
// 					superpixel_people[iter->first].copyTo(NeedToShow[mouse_index]);
// 					mouse_index++;
// 					if(mouse_index > 7)
// 						mouse_index = 0;
// 				}	
// 			}
			
// 		}
// 	}
// } 


int Cost::video_preflow(cv::Mat src)
{
	cv::Mat fgmask;
	bg_model->apply(src, fgmask, update_bg_model ? -1 : 0);
	return 0;
}


cv::Mat Cost::cal_dis(cv::Mat flow_uv0, cv::Mat flow_uv1)
{
	cv::Mat out(flow_uv0.rows, flow_uv0.cols, flow_uv0.type());
	for (int i = 0; i < flow_uv0.rows; i++)
	{
		float* data1 = flow_uv0.ptr<float>(i);
		float* data2 = flow_uv1.ptr<float>(i);
		float* data3 = out.ptr<float>(i);
		for (int j = 0; j < flow_uv0.cols*3; j++)
		{
			/*if (data1[j] * data1[j] + data2[j] * data2[j] > 0.5)
			continue;*/
			data3[j] = data1[j] * data1[j] + data2[j] * data2[j];
		}
	}
	return out;
}

////////set image to m rows and n cols
////////from left to right save
int Cost::SetBlock(cv::Mat img)
{
	for (int i = 0; i < rows_b; i++)
	{
		for (int j = 0; j < cols_b; j++)
		{
			cv::Rect rect(j * img.cols / cols_b, i * img.rows / rows_b,
				img.cols / cols_b, img.rows / rows_b);
			img_rect.push_back(rect);
		}
	}
	return 0;
}

int Cost::Thtracking()
{
	while (1)
	{
		cv::Mat local_bayer, ref_bayer;
		cv::Mat watching;
		std::vector<cam::Imagedata> imgdatas(2);
		cameraPtr->captureFrame(imgdatas);
		cv::Mat(camInfos[0].height, camInfos[0].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[0].data)).copyTo(local_bayer);
		cv::Mat(camInfos[1].height, camInfos[1].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[1].data)).copyTo(ref_bayer);
		cv::Mat local, ref;
		//////////////convert/////////////
		//cv::cvtColor(local_bayer, local, CV_BayerRG2BGR);
		cv::cvtColor(ref_bayer, ref, CV_BayerRG2BGR);
		std::vector<cv::Mat> channels(3);
		/*split(local, channels);
		channels[0] = channels[0] * camInfos[0].blueGain;
		channels[1] = channels[1] * camInfos[0].greenGain;
		channels[2] = channels[2] * camInfos[0].redGain;
		merge(channels, local);*/

		split(ref, channels);
		channels[0] = channels[0] * camInfos[1].blueGain;
		channels[1] = channels[1] * camInfos[1].greenGain;
		channels[2] = channels[2] * camInfos[1].redGain;
		merge(channels, ref);

		tracking(ref);
		thread_flag = 1;

	}
}

void Cost::startTh()
{
	std::thread Th_tracking(&Cost::Thtracking, this);
	Th_tracking.detach();
}

bool Cost::iscontain(cv::Rect roi)
{
	if (tracking_roi.x > roi.x && tracking_roi.y > roi.y
		&& tracking_roi.x + tracking_roi.width < roi.x + roi.width
		&& tracking_roi.y + tracking_roi.height < roi.y + roi.height)
		return 1;
	return 0;
}

int Cost::init_people_detection(cv::Mat img)
{
	std::string cfgfile = "/home/lgh/Documents/gimbleMulti/YOLO/yolov3.cfg";
	std::string weightfile = "/home/lgh/Documents/gimbleMulti/YOLO/yolov3.weights";
	detector_people->init(cfgfile, weightfile);
	detector->init(cfgfile, weightfile);
	SetBlock(img);
	return 0;
}

int Cost::init_face_detection()
{
	std::string cfgfile = "/home/lgh/Documents/gimbleMulti/YOLO/yolo-face.cfg";
	std::string weightfile = "/home/lgh/Documents/gimbleMulti/YOLO/yolo-face.weights";
	detector_face->init(cfgfile, weightfile, 0.1);
	return 0;
}

//according to the ref tracking, to choose the people in local
cv::Mat Cost::SetFaceBlock(cv::Mat local,cv::Mat ref, cv::Point current_point)
{
	cv::Rect current_roi;
	for(int i = 0; i < mytracker.tracks.size(); i++)
	{
		if(mytracker.tracks[i].track_id == current_id)
		{
			cv::Rect rect(mytracker.tracks[i].to_tlwh()(0),mytracker.tracks[i].to_tlwh()(1),
					mytracker.tracks[i].to_tlwh()(2),mytracker.tracks[i].to_tlwh()(3)); 
			current_roi = rect;
			break;
		}
		if(i == mytracker.tracks.size() - 1)
		{
			find_face = 0;
			std::cout<<"tracking lost!!!!!!!!!!!!!!!!!"<<std::endl;
			return local;
		}
	}
	
	
	//cv::Rect current_roi = tracking_roi;   //output from deepsort
	std::vector<cv::Rect> rois = detector_people->detect(local);
	if(rois.size() == 0)
	{
		rois.push_back(cv::Rect(1000,750,200,100));
	}
	std::cout << "in local, we detect " << rois.size() << " people!!" << std::endl;
	cv::Mat output;
	////get nearest people
	int min = 10000;
	int index = 0;
	for (int i = 0; i < rois.size(); i++)
	{
		cv::Point temp;
		temp.x = current_roi.x - current_point.x;
		temp.y = current_roi.y - current_point.y;
		if (abs(temp.x * 8 - rois[i].x) + abs(temp.y * 8 - rois[i].y) < min)
		{
			min = abs(temp.x * 8 - rois[i].x) + abs(temp.y * 8 - rois[i].y);
			index = i;
		}
	}
	//detect the face
	cv::Rect rect(rois[index].x, rois[index].y, rois[index].width, rois[index].height);
	if (rect.x - 50 > 0 && rect.y - 50 > 0
		&& rect.x + rect.width + 50 < local.cols - 1
		&& rect.y + rect.height + 50 < local.rows - 1)
	{
		rect.x -= 50;
		rect.y -= 50;
		rect.width += 100;
		rect.height += 100;
	}

	rect.height /= 2;
	cv::Mat img;
	local(rect).copyTo(img);
	std::vector<cv::Rect> faces = detector_face->detect(img);
	std::cout<<"faces size is          "<<faces.size()<<std::endl;
	cv::Mat out;
	if(faces.size() == 0)
	{
		img.copyTo(out);
		find_face = 0;
	}
	else
	{
		local(rect).copyTo(out);
		find_face = 1;
	}

	return out;
}

//////detect the face according to the detected people
int Cost::face_detection(cv::Mat local,cv::Mat ref,cv::Point current_point)
{
	cv::Mat Vec_people = SetFaceBlock(local,ref, current_point); //�ҵ����п��ܵ����˿�
	if (find_face == 0)
	{
		//is_face[current_id] = 0;
		return 0;
	}
	else
	{
		//Vec_people.copyTo(current_show[0]);
		cv::Mat temp;
		Vec_people.copyTo(temp);
		add_id(current_id, temp);
		temp.copyTo(superpixel_people[current_id]);
		if(NeedToShow.size() < showSub)
		{
			cv::resize(temp,temp,cv::Size(600,600));
			NeedToShow.push_back(temp);
		}
		else
		{
			cv::resize(temp,temp,cv::Size(600,600));
			temp.copyTo(NeedToShow[NeedToShow_index++]);
			if(NeedToShow_index >= showSub)
				NeedToShow_index = 0;
		}
		is_face_id.push_back(current_id);
		find_face = 0;
		return 0;
	}

	std::cout<<"face finish......................"<<std::endl;
}

void Cost::add_id(int current_id, cv::Mat img)
{
	std::string text = std::to_string(current_id);
	int baseline;
	cv::Size text_size = cv::getTextSize(text, CV_FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
	cv::putText(img, text, cv::Point(0, text_size.height), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 0), 2);
	
}

double Cost::people_match(cv::Mat img1, cv::Mat img2)
{
	calib::FeatureMatch match;
	cv::Mat temp1, temp2;
	img1.copyTo(temp1);
	img2.copyTo(temp2);
	cv::resize(temp1, temp1, cv::Size(temp2.cols, temp2.rows));
	double confidence = match.match_people(temp1, temp2);
	return confidence;
}

//int Cost::test_tracking()
//{
//
//}

void Cost::tracking(cv::Mat img)
{
	std::vector<cv::Mat>frame_seg(set_b);
	DETECTIONS detections;
	cv::Mat frame1,frame2;
	img(img_rect[0]).copyTo(frame1);
	img(img_rect[1]).copyTo(frame2);
	
	//only support two rect left and right temporarily
	detector->getFrameDetections(frame1, detections, 0);
	detector->getFrameDetections(frame2, detections, 1);
	detector->dataMoreConf(args_min_confidence, detections);
	detector->dataPreprocessing(args_nms_max_overlap, detections);
	FeatureTensor::getInstance()->getRectsFeature(frame1, frame2, detections);
	
	mytracker.predict();
	mytracker.update(detections);
	std::vector<RESULT_DATA> result;
	for(Track& track : mytracker.tracks) 
	{
		if(!track.is_confirmed() || track.time_since_update > 1) continue;
		result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
	}

	char showMsg[10];

	// for(unsigned int k = 0; k < detections.size(); k++) 
	// {
	// 	DETECTBOX tmpbox = detections[k].tlwh;
	// 	if(detections[k].aa == 0)
	// 	{
	// 		cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
	// 		cv::rectangle(frame1, rect, cv::Scalar(0,0,255), 4);
	// 	}
				
	// 	if(detections[k].aa == 1)
	// 	{
	// 		cv::Rect rect(tmpbox(0) - frame1.cols, tmpbox(1), tmpbox(2), tmpbox(3));	
	// 		cv::rectangle(frame2, rect, cv::Scalar(0,0,255), 4);
	// 	}
				
	// }


	current_tracking.clear();
	finish_push = 0;
	for(unsigned int k = 0; k < result.size(); k++) 
	{
		DETECTBOX tmp = result[k].second;
		if(tmp(0) >= frame1.cols)
    	{
    		tmp(0) -= frame1.cols;
			cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
			if(result[k].first > max_cap)
			{
				result[k].first = result[k].first % max_cap;
			}

			current_tracking[result[k].first] = cv::Rect(rect.x + frame1.cols, rect.y, rect.width, rect.height);

			sprintf(showMsg, "%d", result[k].first);
			
			std::vector<int>::iterator iter = find(is_face_id.begin(),is_face_id.end(),result[k].first);
			if(iter == is_face_id.end())  
			{
				cv::rectangle(frame2, rect, cv::Scalar(255, 255, 0), 2);
				cv::putText(frame2, showMsg, cv::Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 4);
			}
			else
			{
				cv::rectangle(frame2, rect, cv::Scalar(255, 0, 0), 2);
				cv::putText(frame2, showMsg, cv::Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 4);
			}

			
    	}
		else
		{
			cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
			if(result[k].first > max_cap)
			{
				result[k].first = result[k].first % max_cap;
			}

			current_tracking[result[k].first] = rect;

			sprintf(showMsg, "%d", result[k].first);

			std::vector<int>::iterator iter = find(is_face_id.begin(),is_face_id.end(),result[k].first);
			if(iter == is_face_id.end()) 
			{
				cv::rectangle(frame1, rect, cv::Scalar(255, 255, 0), 2);
				cv::putText(frame1, showMsg, cv::Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 4);
			}
			else
			{
				cv::rectangle(frame1, rect, cv::Scalar(255, 0, 0), 2);
				cv::putText(frame1, showMsg, cv::Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 4);
			}

			
		}
		
	}
	finish_push = 1;
	frame1.copyTo(frame_seg[0]);
	frame2.copyTo(frame_seg[1]);
		
	merge_img(frame_seg).copyTo(show_opencv);
	
}

cv::Mat Cost::merge_img(std::vector<cv::Mat>frame_seg)
{
	int row = rows_b * frame_seg[0].rows;
	int col = cols_b * frame_seg[0].cols;
	cv::Mat img(row,col,CV_8UC3);
	for(int i=0;i<rows_b;i++)
	{
		for(int j=0;j<cols_b;j++)
		{
			cv::Rect rect(j * frame_seg[0].cols, i * frame_seg[0].rows, frame_seg[0].cols, frame_seg[0].rows);
			frame_seg[i * cols_b + j].copyTo(img(rect));
		}
	}

	return img;
}

int Cost::SeekNextDst(cv::Mat src, cv::Point& dst_point)
{
	int index;
	if(mytracker.tracks.size() == 0)
	{
		std::cout<<"we dont detect any person!!!"<<std::endl;
		return -1;
	}
	for(int i = 0;i < mytracker.tracks.size();i++)
	{
		//std::unordered_map<int, cv::Mat>::iterator result = find(superpixel_people.begin(),superpixel_people.end(),mytracker.tracks[i].track_id);
		//std::unordered_map<int, cv::Mat>::iterator result = superpixel_people.find(mytracker.tracks[i].track_id);
		std::vector<int>::iterator result = find(tracked_id.begin(),tracked_id.end(),mytracker.tracks[i].track_id);
		if(result == tracked_id.end())
		{
			tracked_id.push_back(mytracker.tracks[i].track_id);
			current_id = mytracker.tracks[i].track_id;
			index = i;
			break;
		}
		else
		{
			continue;
		}
	}
	dst_point.x = mytracker.tracks[index].to_tlwh()(0) - static_cast<int>(src.cols*0.092);
	dst_point.y = mytracker.tracks[index].to_tlwh()(1) - static_cast<int>(src.rows*0.092);

	cv::Rect rect(mytracker.tracks[index].to_tlwh()(0),mytracker.tracks[index].to_tlwh()(1),
					mytracker.tracks[index].to_tlwh()(2),mytracker.tracks[index].to_tlwh()(3)); 
	tracking_roi = rect;
	return 0;

}