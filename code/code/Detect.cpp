#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include "Detect.h"
#include<string>
// #include <QDebug.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Initialize the parameters

Detector& Detector::operator=(Detector& detector){
	this->modelConfiguration_m = detector.modelConfiguration_m;
	this->modelBinary_m = detector.modelBinary_m;
	this->nameFile_m = detector.nameFile_m;

	//set model
	this->classNamesVec = detector.classNamesVec ;
	this->net = detector.net;
	this->layerNames = detector.layerNames;
	this->outPutNames = detector.outPutNames;
	this->outLayers = detector.outLayers;
	return *this;
}

Detector::Detector(string modelConfiguration, string modelBinary, string nameFile){
		
		this->modelConfiguration_m =  modelConfiguration;
		this->modelBinary_m = modelBinary;
		this->nameFile_m = nameFile;
		this->net = readNetFromDarknet(this->modelConfiguration_m, this->modelBinary_m);
		// cout<<net<<endl;
		if (net.empty())
		{
			printf("Could not load net...\n");
		}

		ifstream classNamesFile(nameFile);
		if (classNamesFile.is_open())
		{
			string className = "";
			while (std::getline(classNamesFile, className))
				this->classNamesVec.push_back(className);
		}


		this->layerNames= net.getLayerNames();
		this->outLayers = net.getUnconnectedOutLayers();

		for(int index = 0; index < outLayers.size(); index++)
		{
			
		    this->outPutNames.push_back(layerNames[outLayers[index] - 1].c_str());
		    // qDebug() << __FILE__ << __LINE__<< QString(layerNames[outLayers[index] - 1].c_str());
		}


}

vector<vector<float>> Detector::detect(Mat frame, float confidenceThreshold, int X_start, int Y_start){
	vector<vector<float>> reslut;
	ostringstream ss;
	Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), false, false);
	this->net.setInput(inputBlob, "data");
	std::vector<cv::Mat> probs;
	net.forward(probs, outPutNames); // start detect 
	for(int index = 0; index < probs.size(); index++)
	{
		// cout<<probs[index]<<endl;
		for (int i = 0; i < probs[index].rows; i++)
		{
			const int probability_index = 5;
			const int probability_size = probs[index].cols - probability_index;
			float *prob_array_ptr = &probs[index].at<float>(i, probability_index);
			size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
			// cout<<objectClass<<endl;
			float confidence = probs[index].at<float>(i, (int)objectClass + probability_index);

			if (confidence >= confidenceThreshold)
			{
				float x = probs[index].at<float>(i, 0);
				float y = probs[index].at<float>(i, 1);
				float width = probs[index].at<float>(i, 2);
				float height = probs[index].at<float>(i, 3);
				float xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
				float yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
				float xRightTop = static_cast<int>((x + width / 2) * frame.cols);
				float yRightTop = static_cast<int>((y + height / 2) * frame.rows);
				// reslut.push_back(to_string(objectClass)+" "+to_string(confidence)+" "+to_string(xLeftBottom + X_start)+" "+to_string(yLeftBottom + Y_start)+" "+to_string(xRightTop + X_start)+" "+to_string(yRightTop + Y_start));
				reslut.push_back({objectClass,confidence,xLeftBottom+ X_start,yLeftBottom + Y_start,xRightTop+ X_start,yRightTop + Y_start});
				Rect object(xLeftBottom, yLeftBottom,
					xRightTop - xLeftBottom,
					yRightTop - yLeftBottom);
				
				// cout<<object<<endl;
				// cout<<"debug"<<endl;
				rectangle(frame, object, Scalar(255, 0, 255), 2, 3);
				if (objectClass < classNamesVec.size())
				{

					ss.str("");
					ss << confidence;
					String conf(ss.str());
					String label = String(classNamesVec[objectClass]) + ": " + conf;
					int baseLine = 0;
					Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
					rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom),
						Size(labelSize.width, labelSize.height + baseLine)),
						Scalar(255, 255, 255), CV_FILLED);
					putText(frame, label, Point(xLeftBottom, yLeftBottom + labelSize.height),
						FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
				}
			}
		}
	}
	// cout<<"detecting..."<<endl;
	// imshow("YOLO-Detections", frame);
	// for (int i = 0; i < reslut.size(); ++i)
	// {
	// 	for (int j = 0; j <reslut[i].size() ; ++j)
	// 	{
	// 		cout<<reslut[i][j]<<" ";
	// 	}
	// 	cout<<endl;
	// }
	

	// waitKey(100);

	return reslut;
}




