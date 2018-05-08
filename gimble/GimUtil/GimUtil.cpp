#include "GimUtil.h"
#include <time.h>

GimUtil::GimUtil(){}
GimUtil::~GimUtil(){}

bool GimUtil::isInside(cv::Point2f pt, cv::Rect rect) {
	if (rect.contains(pt))
		return true;
	else return false;
}

cv::Mat GimUtil::SEDDetector(cv::Mat img, float scale) {
	cv::Mat edgeImg;
	cv::Size size_large = img.size();
	cv::Size size_small = cv::Size(size_large.width * scale, size_large.height * scale);
	cv::resize(img, img, size_small);
	img.convertTo(img, cv::DataType<float>::type, 1 / 255.0);
	ptr->detectEdges(img, edgeImg);
	edgeImg = edgeImg * 255;
	cv::resize(edgeImg, edgeImg, size_large);
	edgeImg.convertTo(edgeImg, CV_8U);
	return edgeImg;
}

int GimUtil::colorCorrectRGB(cv::Mat & srcImg, cv::Mat dstImg) 
{
	cv::Scalar meanSrc, stdSrc, meanDst, stdDst;
	cv::meanStdDev(srcImg, meanSrc, stdSrc);
	cv::meanStdDev(dstImg, meanDst, stdDst);
	std::vector<cv::Mat> channel;
	cv::split(srcImg, channel);
	for (int i = 0; i < 3; i++)
	{
		channel[i].convertTo(channel[i], -1, stdDst.val[i] / stdSrc.val[i],
			meanDst.val[i] - stdDst.val[i] / stdSrc.val[i] * meanSrc.val[i]);
	}
	cv::merge(channel, srcImg);
	
	return 0;
}

int GimUtil::find_position(cv::Mat refImg, cv::Mat localImg, cv::Point &out_point)
{
	cv::Mat refEdge = this->SEDDetector(refImg, 0.5);
	cv::Mat localEdge = this->SEDDetector(localImg, 0.5);
	// resize localview image
	cv::Mat templ, templEdge;
	sizeBlock = cv::Size(localImg.cols * scale, localImg.rows * scale);
	cv::resize(localImg, templ, sizeBlock);
	cv::resize(localEdge, templEdge, sizeBlock);

	//colorCorrectRGB(templ, refImg);    ///////////////////////
	cv::Mat result, resultEdge;
	cv::matchTemplate(refImg, templ, result, cv::TM_CCOEFF_NORMED);
	cv::matchTemplate(refEdge, templEdge, resultEdge, cv::TM_CCOEFF_NORMED);
	result = result.mul(resultEdge);

	cv::Point maxLoc;
	cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	out_point = maxLoc;

	return 0;
}

int GimUtil::gimble_find_position(cv::Mat refImg, cv::Mat localImg, cv::Point ref_point, 
	                               float region_mul,cv::Point &out_point)     //region_mul must bigger than 1,reginal:243*182
{
	int region_width, region_height;
	cv::Rect imgrect(0, 0, refImg.cols, refImg.rows);
	cv::Size origin_size(refImg.cols*scale, refImg.rows*scale);
	cv::Size new_size(refImg.cols*scale*region_mul, refImg.rows*scale*region_mul);
	cv::Point2f new_point = cv::Point2f(ref_point.x - (new_size.width - origin_size.width) / 2,
		ref_point.y - (new_size.height - origin_size.height) / 2);
	cv::Rect new_region;
	cv::Mat new_ref;

	if (isInside(new_point, imgrect) && isInside(cv::Point2f(new_point.x+new_size.width, new_point.y + new_size.height), imgrect))
	{
		new_region=cv::Rect(new_point, new_size);
	}
	else
	{
		out_point = ref_point;
		return -1;
	}
	refImg(new_region).copyTo(new_ref);

	
	cv::Mat refEdge = this->SEDDetector(new_ref, 0.5);
	cv::Mat localEdge = this->SEDDetector(localImg, 0.5);
	// resize localview image
	cv::Mat templ, templEdge;
	sizeBlock = cv::Size(localImg.cols * scale, localImg.rows * scale);
	cv::resize(localImg, templ, sizeBlock);
	cv::resize(localEdge, templEdge, sizeBlock);
	//colorCorrectRGB(templ, new_ref);    ///////////////////////

	cv::Mat result, resultEdge;
	cv::matchTemplate(new_ref, templ, result, cv::TM_CCOEFF_NORMED);
	cv::matchTemplate(refEdge, templEdge, resultEdge, cv::TM_CCOEFF_NORMED);
	result = result.mul(resultEdge);

	cv::Point maxLoc;
	cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	out_point.x = maxLoc.x + new_point.x;
	out_point.y = maxLoc.y + new_point.y;

	std::cout << "This current x is:" << out_point.x << "    y is:" << out_point.y << std::endl;

	return 0;
}

int GimUtil::init_stitcher()
{
	scale = 0.118;
	ptr = cv::ximgproc::createStructuredEdgeDetection("/home/lgh/Documents/gimbleMulti/model/model.yml");
	return 0;
}

int GimUtil::move(cv::Point& dst)       //�������еĵ�ǰ��͵�ǰ����ʹ��̨�ƶ���Ŀ������λ��
{
	float pulse_delta;
	cv::Point2f dst_pulse;
	pulse_delta = (dst.y - current_point.y) / 3.0;
	dst_pulse.y = current_pulse.y - pulse_delta;
	current_pulse.y = dst_pulse.y;

	pulse_delta = (dst.x - current_point.x) / 4.0;  //3.0
	dst_pulse.x = current_pulse.x - pulse_delta;
	current_pulse.x = dst_pulse.x;

	std::cout << current_pulse << std::endl;
	current_point.x = dst.x;
	current_point.y = dst.y;
	return 0;
}

int GimUtil::detect_move(cv::Point dst)       //�������еĵ�ǰ��͵�ǰ����ʹ��̨�ƶ���Ŀ������λ��
{
	float pulse_delta;
	cv::Point2f dst_pulse;
	pulse_delta = (dst.y - current_point.y) / 3.0;
	dst_pulse.y = current_pulse.y - pulse_delta;
	if (dst_pulse.y >= PM + 120 - dY)    //��̨ת������ȫ��ͼ��Χ
	{
		return -1;
		std::cout << "dst_pulse.y >= PM + 120" << std::endl;
	}
	else if (dst_pulse.y <= PM + 120 - dY*(Row - 1))
	{
		return -1;
		std::cout << "dst_pulse.y <= PM + 120 - dY*Row" << std::endl;
	}

	pulse_delta = (dst.x - current_point.x) / 4.0;  //3.0
	dst_pulse.x = current_pulse.x - pulse_delta;
	if (dst_pulse.x >= YM + 180 - dX)
	{
		return -1;
		std::cout << "dst_pulse.x >= YM + 180" << std::endl;
	}
	else if (dst_pulse.x <= YM + 180 - dX*(Col - 1))
	{
		return -1;
		std::cout << "dst_pulse.x <= YM + 180 - dX*Col" << std::endl;
	}
	return 0;
}