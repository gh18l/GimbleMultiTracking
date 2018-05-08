#ifndef __YOLODETECT_HPP___
#define __YOLODETECT_HPP___

#include <iostream>
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class YOLODetect{
public:
    YOLODetect();
    ~YOLODetect();
    float thresh;
    char **names;
    char *input;
    float nms;
private:
    std::shared_ptr<void> detector_ptr;
    static void YOLOrgbgr_image(image im);
    static float *YOLOnetwork_predict(network net, float *input);
    static detection *YOLOget_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
    static void YOLOdo_nms_sort(detection *dets, int total, int classes, float thresh);
    static int YOLOmax_index(float *a, int n);
    static void YOLOfree_image(image m);
    void YOLOinit(std::string cfgfile, std::string weightfile,std::string datacfg);
}


#endif