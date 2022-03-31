#pragma once
#include "Detect.h"
#include "Utils.h"
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<string>

using namespace cv;
using namespace std;

class Detector_big_image{
	int x_start, x_end;
	int y_start, y_end;
	int cut_x, cut_y;
	int stride;
	float thresh_m;
	string modelConfiguration;
	string modelBinary;
	string nameFile;
	// Detector detector(modelConfiguration, modelBinary, nameFile);
	// there has a question to think
public:
	Detector_big_image(float thresh, int stride_m, int cut_x, int cut_y, string modelConfiguration, string modelBinary, string nameFile);
	vector<vector<float>> detect(Mat &image);
};