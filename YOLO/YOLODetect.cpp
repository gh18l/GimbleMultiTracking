#include "YOLODetect.hpp"

#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "network.h"


#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
YOLODetect::YOLODetect():thresh(0.3){}
YOLODetect::~YOLODetect(){}

struct YOLODetetct_ptr{
    layer l;
    network net;
    list *options;
    detection *dets;
    box b;
}

void YOLODetect::YOLOrgbgr_image(image im)
{
    rgbgr_image(im);
}

float *YOLODetect::YOLOnetwork_predict(network net, float *input)
{
    return network_predict(net, input);
}

detection *YOLODetect::YOLOget_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
    return get_network_boxes(net, w, h, thresh, hier, map, relative, num, letter);
}

void YOLODetect::YOLOdo_nms_sort(detection *dets, int total, int classes, float thresh)
{
    do_nms_sort(dets, total, classes, thresh);
}

int YOLODetect::YOLOmax_index(float *a, int n)
{
    return max_index(a, n);
}

void YOLODetect::YOLOfree_image(image m)
{
    free_image(m);
}

void YOLODetect::YOLOinit(std::string cfgfile, std::string weightfile,std::string datacfg)
{
    detector_ptr = std::make_shared<YOLODetetct_ptr>();
    std::shared_ptr<YOLODetetct_ptr> detector = std::static_pointer_cast<YOLODetetct_ptr>(detector_ptr); 
    detector->options = read_data_cfg(const_cast<char*>(datacfg.c_str()));
    char *name_list = option_find_str(detector->options, "names", "data/names.list");
    names = get_labels(name_list);

    detector->net = parse_network_cfg_custom(const_cast<char*>(cfgfile.c_str()), 1);
    if(weightfile){
		load_weights(&(detector->net), const_cast<char*>(weightfile.c_str()));
	}

    set_batch_network(&(detector->net), 1);
    srand(2222222);

    char buff[256];
	input = buff;
	nms=.4;
}

