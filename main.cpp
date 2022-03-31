#include "code/Detect.h"
#include "code/Detector.h"
#include<iostream>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>
#include"code/Utils.h"

using namespace std;
using namespace cv;


void test(){
	string modelConfiguration = "/home/wzq/WorkPlace/yolo(C++)/yolo/cfg/yolov4_0722.cfg";
	string modelBinary = "/home/wzq/WorkPlace/yolo(C++)/yolo/weights/yolov4_0722_best.weights";
	string nameFile = "/home/wzq/WorkPlace/yolo(C++)/yolo/data/data.names";
	string filepath = "/home/wzq/WorkPlace/yolo(C++)/yolo/resluts/input.jpg";
	string imageName = "/home/wzq/WorkPlace/yolo(C++)/yolo/images/input.jpg";
	Mat image = imread(imageName, 0); 
	Detector detector(modelConfiguration, modelBinary, nameFile);
	vector<vector<float>> result;
	
	result = detector.detect(image, 0.6, 0, 0); // Mat image , confidence, x_start , y_start

}

void run(){
	string txt_save_root = "/home/wzq/WorkPlace/yolo(C++)/yolo/txts/";
	string image_root = "/home/wzq/WorkPlace/yolo(C++)/yolo/images/";//"/home/wzq/SARdata/Images/";
	string imgae_save_path = "/home/wzq/WorkPlace/yolo(C++)/yolo/resluts/";



	string modelConfiguration = "/home/wzq/WorkPlace/yolo(C++)/yolo/cfg/yolov4_0722.cfg";
	string modelBinary = "/home/wzq/WorkPlace/yolo(C++)/yolo/weights/yolov4_0722_best.weights";
	string nameFile = "/home/wzq/WorkPlace/yolo(C++)/yolo/data/data.names";
	string filepath = "/home/wzq/WorkPlace/yolo(C++)/yolo/resluts/input.jpg";
	// string imageName = "/home/wzq/WorkPlace/yolo(C++)/yolo/images/264.jpg";

	Detector_big_image detector(0.7, 300, 416, 416,modelConfiguration, modelBinary, nameFile); // inital detector
	vector<string> names = getFiles(image_root);
	cout<<"there has "<<names.size()<<" images ."<<endl;
	for (int i = 0; i < names.size(); ++i)
	{
		string imageName = image_root+names[i];
		cout<<imageName<<endl;
		Mat image = imread(imageName, 0);
		vector<vector<float>> result = detector.detect(image);

		cout<<"there has "<<result.size()<<endl;
		string::size_type pos = names[i].find(".");
		// saveTxt(txt_save_root, names[i].substr(0,pos) + ".txt", result);
		
		// draw(image, result, imgae_save_path+names[i], nameFile);
	}
	 

	


}

int main(int argc, char const *argv[])
{
	// test();
	run();

	return 0;
}


