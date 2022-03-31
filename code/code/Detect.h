#pragma once

#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

class Detector{
	//read model
	string modelConfiguration_m;
	string modelBinary_m;
	string nameFile_m;

	//set model
	vector<string> classNamesVec;
	dnn::Net net;
	vector<String> layerNames;
	std::vector<cv::String> outPutNames;
	std::vector<int> outLayers;
public:
	Detector& operator=(Detector& detector); // overload = 
	Detector(string modelConfiguration, string modelBinary,string nameFile);

	vector<vector<float>> detect(Mat image, float confidenceThreshold, int X_start,int Y_start);

};
