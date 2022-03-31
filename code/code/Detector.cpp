#include "Detector.h"
#include "Detect.h"
#include "Utils.h"
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<string>
using namespace cv;
using namespace std;

Detector_big_image::Detector_big_image(float thresh,int stride_m, int cut_x_m, int cut_y_m, string modelConfiguration_m, string modelBinary_m, string nameFile_m){
	this->x_start=0;
	this->x_end=0;
	this->y_start=0;
	this->y_end=0;
	this->cut_x=cut_x_m;
	this->cut_y=cut_y_m;
	this->stride = stride_m;
	this->thresh_m = thresh;
	// *this->detector = Detector(modelConfiguration, modelBinary, nameFile);
	this->modelConfiguration = modelConfiguration_m;
	this->modelBinary = modelBinary_m;
	this->nameFile = nameFile_m;

}

vector<vector<float>> Detector_big_image::detect(Mat &image){
	Detector detector(modelConfiguration, modelBinary, nameFile);
	image = image.clone();
	int row = image.rows;
	int col = image.cols;
	Mat target;
	Mat sub_img;
	vector<vector<float>> result;
	vector<vector<float>> output;
	for (int y = 0; y < row; y+=stride)
	{
		for (int x = 0; x < col; x+=stride)
		{
			x_start = x;
			y_start = y;
			if(x_start + cut_x >= col){
				x_start = col - cut_x - 1;
			}
			if(y_start + cut_y >= row){
				y_start = row - cut_y - 1;
			}

			sub_img = image(Rect(x_start,y_start,cut_x,cut_y)).clone();
			output = detector.detect(sub_img, thresh_m, x_start, y_start);
			
			for (int i = 0; i < output.size(); ++i)
			{
				if((int(output[i][0])==0 && output[i][1]>=0.91) || (int(output[i][0])==1 && output[i][1]>=0.91)){ // set threash
					bool car = is_car(output[i], image);

					result.push_back(output[i]);
				}
			}
			output.clear();

		}
	}
	cout<<result.size()<<endl;
	
	return nms(result);
}
