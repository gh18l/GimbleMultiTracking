/*
 * FeatureTensor.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: zy
 */

#include "FeatureTensor.h"
using namespace tensorflow;

#define TENSORFLOW_MODEL_META "RUNNINGDATA/tensor_networks/111.meta"
#define TENSORFLOW_MODEL "RUNNINGDATA/tensor_networks/mars-small128.ckpt-68577"

FeatureTensor *FeatureTensor::instance = NULL;

FeatureTensor *FeatureTensor::getInstance() {
	if(instance == NULL) {
		instance = new FeatureTensor();
	}
	return instance;
}

FeatureTensor::FeatureTensor() {
	//prepare model:
	bool status = init();
	if(status == false) exit(1);
}

FeatureTensor::~FeatureTensor() {
	session->Close();
	delete session;
	output_tensors.clear();
	outnames.clear();
}

bool FeatureTensor::init() {
	tensorflow::SessionOptions sessOptions;
	sessOptions.config.mutable_gpu_options()->set_allow_growth(true);
	session = NewSession(sessOptions);
	if(session == nullptr) return false;

	const tensorflow::string pathToGraph = TENSORFLOW_MODEL_META;
	Status status;
	MetaGraphDef graph_def;
	status = ReadBinaryProto(tensorflow::Env::Default(), pathToGraph, &graph_def);
	if(status.ok() == false) return false;

	status = session->Create(graph_def.graph_def());
	if(status.ok() == false) return false;

	const tensorflow::string checkpointPath = TENSORFLOW_MODEL;
	Tensor checkpointTensor(DT_STRING, TensorShape());
	checkpointTensor.scalar<std::string>()() = checkpointPath;
	status = session->Run(
			{ {graph_def.saver_def().filename_tensor_name(), checkpointTensor}, },
			{}, {graph_def.saver_def().restore_op_name()}, nullptr );
	if(status.ok() == false) return false;

	input_layer = "Placeholder:0";
	outnames.push_back("truediv:0");
	feature_dim = 128;
	return true;
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img1, const cv::Mat& img2, DETECTIONS& d) {
	std::vector<cv::Mat> mats;
	for(DETECTION_ROW& dbox : d) {
		if(dbox.aa == 0)
		{
			cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
				int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
		rc.width = rc.height * 0.5;
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img1.cols? rc.width: (img1.cols-rc.x));
		rc.height = (rc.y + rc.height <= img1.rows? rc.height:(img1.rows - rc.y));

		cv::Mat mattmp = img1(rc).clone();
		cv::resize(mattmp, mattmp, cv::Size(64, 128));
		mats.push_back(mattmp);
		}
		if(dbox.aa == 1)
		{
			cv::Rect rc = cv::Rect(int(dbox.tlwh(0) - img1.cols), int(dbox.tlwh(1)),
				int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
		rc.width = rc.height * 0.5;
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img2.cols? rc.width: (img2.cols-rc.x));
		rc.height = (rc.y + rc.height <= img2.rows? rc.height:(img2.rows - rc.y));

		cv::Mat mattmp = img2(rc).clone();
		cv::resize(mattmp, mattmp, cv::Size(64, 128));
		mats.push_back(mattmp);
		}
			
		
	}
	int count = mats.size();

	Tensor input_tensor(DT_UINT8, TensorShape({count, 128, 64, 3}));
	tobuffer(mats, input_tensor.flat<uint8>().data());
	std::vector<std::pair<tensorflow::string, Tensor>> feed_dict = {
			{input_layer, input_tensor},
	};
	Status status = session->Run(feed_dict, outnames, {}, &output_tensors);
	if(status.ok() == false) return false;
	float* tensor_buffer = output_tensors[0].flat<float>().data();
	int i = 0;
	for(DETECTION_ROW& dbox : d) {
		if(dbox.aa == 0)
		{
			for(int j = 0; j < feature_dim; j++)
			dbox.feature[j] = tensor_buffer[i*feature_dim+j];
			i++;
		}
		if(dbox.aa == 1)
		{
			for(int j = 0; j < feature_dim; j++)
			dbox.feature[j] = tensor_buffer[i*feature_dim+j];
			i++;
		}
		
	}
	return true;
}

void FeatureTensor::tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf) {
	int pos = 0;
	for(const cv::Mat& img : imgs) {
		int Lenth = img.rows * img.cols * 3;
		int nr = img.rows;
		int nc = img.cols;
		if(img.isContinuous()) {
			nr = 1;
			nc = Lenth;
		}
		for(int i = 0; i < nr; i++) {
			const uchar* inData = img.ptr<uchar>(i);
			for(int j = 0; j < nc; j++) {
				buf[pos] = *inData++;
				pos++;
			}
		}//end for
	}//end imgs;
}
