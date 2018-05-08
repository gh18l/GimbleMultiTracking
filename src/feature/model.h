#ifndef MODEL_H
#define MODEL_H
#include "dataType.h"
#include <map>
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
#include "network.h"
#include <string>
/**
 * Each rect's data structure.
 * tlwh: topleft point & (w,h)
 * confidence: detection confidence.
 * feature: the rect's 128d feature.
 */
class DETECTION_ROW {
public:
    DETECTBOX tlwh; //np.float
    float confidence; //float
    FEATURE feature; //np.float32
    DETECTBOX to_xyah() const;
    DETECTBOX to_tlbr() const;
    int aa;
};

typedef std::vector<DETECTION_ROW> DETECTIONS;

/**
 * Get each image's rects & corresponding features.
 * Method of filter conf.
 * Method of preprocessing.
 */
class ModelDetection
{

public:
    void init(std::string cfgfile, std::string weightfile,float thr = 0.3);
    bool loadDataFromFile(const char* motDir, bool withFeature);
    bool getFrameDetections(int frame_idx, DETECTIONS& res);
    bool getFrameDetections(cv::Mat& frame, DETECTIONS& res, int aa);
    bool getFrameDetections(cv::Mat& frame, DETECTIONS& res);
    void dataMoreConf(float min_confidence, DETECTIONS& d);
    void dataPreprocessing(float max_bbox_overlap, DETECTIONS& d);
    std::vector<cv::Rect> detect(cv::Mat frame);
    ModelDetection();
    ~ModelDetection();

private:
    ModelDetection(const ModelDetection&);
    ModelDetection& operator =(const ModelDetection&);
    static ModelDetection* instance;

    using AREAPAIR = std::pair<int, double>;
    struct cmp {
        bool operator()(const AREAPAIR a, const AREAPAIR b) {
            return a.second < b.second;
        }
    };
    std::map<int, DETECTIONS> data;
    void _Qsort(DETECTIONS d, std::vector<int>& a, int low, int high);
    bool loadFromFile;

    //darknet:
    char *input;
    network net;
    clock_t time;
    float thresh;
    float nms;
    char **names;
    //image **alphabet;

    image ipl_to_image(IplImage* src);
};

#endif // MODEL_H
