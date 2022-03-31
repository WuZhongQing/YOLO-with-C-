#include"Utils.h"
#include<iostream>
// #include<io.h>
#include <sys/io.h>
#include<vector>
#include<fstream>
#include"opencv2/opencv.hpp"
#include<math.h>
#include <dirent.h>
#include<string>


#define PI (3.1415926535)



using namespace std;
using namespace cv;


// for windows
// void getFiles(string path, vector<string>& files)
// {
// 	//ÎÄ¼þ¾ä±ú
// 	intptr_t hFile = 0;
// 	//ÎÄ¼þÐÅÏ¢
// 	struct _finddata_t fileinfo;
// 	string p;

// 	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
// 	{
// 		do
// 		{
// 			//Èç¹ûÊÇÄ¿Â¼,µü´úÖ®
// 			//Èç¹û²»ÊÇ,¼ÓÈëÁÐ±í
// 			if ((fileinfo.attrib & _A_SUBDIR))
// 			{
// 				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
// 					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
// 			}
// 			else
// 			{
// 				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
// 			}
// 		} while (_findnext(hFile, &fileinfo) == 0);
// 		_findclose(hFile);
// 	}
// }

vector<string> getFiles(string root){
	DIR *dp;
	struct dirent *dirp;

	vector<string> filename;

	if((dp = opendir(root.c_str()))==NULL){
		cout<<"failed !"<<endl;
	}
	while ((dirp=readdir(dp))!=NULL)
	{
		string temp = dirp->d_name;
		if(temp != "." && temp != ".."){
			filename.push_back(dirp->d_name);
	}
	}
	closedir(dp);
	
	return filename;
}
int getThreash(const Mat& img, double a) { // read only
	double minv = 0;
	double maxv = 0;
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(img, minp, maxp);
	return (int)maxv * a;//set thresh

}

int findCounterNum(Mat& img) {
	vector<vector<Point> > 	contours;
	vector<Vec4i> 			hierarchy;
	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());// CV_RETR_EXTERNAL
	return contours.size();
}

//im2SIFT º¯Êý ÊäÈë£º»Ò¶ÈÍ¼Æ¬  Êä³ö£ºÒ»¸ö vector<double> 1¡Á128
vector<double>  im2SIFT(const Mat& img) {
	int H = img.rows;
	int W = img.cols;
	int L = max(H, W);
	Mat temp = img;
	Size size(L, L);
	resize(temp, temp, size);// ½«Í¼Æ¬±äÎªÕý·½ÐÎ£¬±ãÓÚ´¦Àí
	temp.convertTo(temp, CV_32FC1); //×ª»»Îª¸¡µãÊý
	int L1 = L / 4;
	vector<double> features(4 * 4 * 8);//128¸öÌØÕ÷
	Mat im_i, Xp1, Xm1, Yp1, Ym1, m(Size(L1-2,L1-2), CV_32FC1);
	double theta;
	int index; int label = 0;

	for (int i = 0; i < 4 * L1; i += L1) {
		for (int j = 0; j < 4 * L1; j += L1) {
			im_i = temp(Rect(i, j, L1, L1)).clone();
			Xp1 = im_i(Rect(2, 1, L1 - 2, L1 - 2)).clone(); //ÒòÎªC++ÊÇ´Ó0¿ªÊ¼µÄ£¬ËùÒÔ±ÈmatlabÉÙ1
			Xm1 = im_i(Rect(0, 1, L1 - 2, L1 - 2)).clone();
			Yp1 = im_i(Rect(1, 2, L1 - 2, L1 - 2)).clone();
			Ym1 = im_i(Rect(1, 0, L1 - 2, L1 - 2)).clone();
			pow(((Xp1 - Xm1).mul(Xp1 - Xm1) + (Yp1 - Ym1).mul(Yp1 - Ym1)),0.5, m); // ¿ªÆ½·½£¬²¢¸³Öµ¸øm //¼ÆËã·ù¶ÈÖµ
			vector<double> feature(8, 0.0);

			float X, Y;
			for (int x = 1; x < L1-1; x++) {
				for (int y = 1; y < L1-1; y++) {
					X = im_i.at<float>(x + 1, y) - im_i.at<float>(x - 1, y);
					Y = im_i.at<float>(x, y + 1) - im_i.at<float>(x, y - 1);
					if (X == 0.0) {
						X = 0.00001;//ÓÃÒ»¸öÐ¡Êý´úÌæ0£¬²»»á±¨´í
					}
					theta = atan2(Y, X) + PI;
					index = round(theta / (PI / 4));
					if (index == 8) {
						index = 7;
					}
					feature[index] = feature[index] + m.at<float>(x - 1, y - 1);
				}
			}
			copy(feature.begin(), feature.end(), features.begin() + label); //µÈ¼ÛÓÚ features((label*8 + 1):(label+1)*8) = feature;
			label += 8;

		}
	}
	return features;

}

//¶ÁÈ¡Êý¾ÝµÄº¯Êý
void readdatas(vector<vector<double>>& datav, bool flag, string& model) {
	char temp;
	vector<double> data;
	double number;
	string num1;
	fstream input;
	int index = 1;
	string txtName;
	if (model == "train") {
		txtName = "train_NegativeFeatures.txt";
		if (flag) {
			txtName = "train_PositiveFeatures.txt";
		}
	}
	if (model == "test") {
		txtName = "test_NegativeFeatures.txt";
		if (flag) {
			txtName = "test_PositiveFeatures.txt";
		}
	}
	input.open(txtName, ios::in);
	cout << "Readding " + txtName << endl;
	while ((temp = input.get()) != EOF) {
		num1 += temp;

		if (temp == ' ' || temp == '\n') {
			number = atof(num1.c_str()); //Êý¾ÝÀàÐÍ×ª»»
			data.push_back(number);
			num1 = "";
		}
		if (temp == '\n') {
			datav.push_back(data);
			data.clear();
		}
	}
}

void find(Mat& image, float num, vector<Point>& points)//¹¤¾ßclass
{
	float* pPixel;
	for (int row = 0; row < image.rows; ++row)
	{
		pPixel = image.ptr<float>(row);
		for (int col = 0; col < image.cols; ++col)
		{
			//cout << *pPixel << endl;;
			if (*pPixel == num)
			{

				points.push_back(Point(col, row));
			}
			++pPixel;
		}
	}
}

float m_mean(Mat& img, vector<Point>& points) {
	float sum = 0;

	for (int i = 0; i < points.size(); i++) {
		sum += img.at<float>(points[i]) / points.size();
	}
	return (float)sum;
}

float m_std(Mat& img, vector<Point>& points, float miu) {
	float sum = 0;
	for (int i = 0; i < points.size(); i++) {
		sum += (img.at<float>(points[i]) - miu) * (img.at<float>(points[i]) - miu);
	}
	float s = sqrt((float)(sum / points.size()));
	return s;
}


void bwareaopen(Mat& src, Mat& dst, double min_area, vector<Rect>& bboxs, int m_padsize) {
	dst = src.clone();
	vector<vector<Point> > 	contours;
	vector<Vec4i> 			hierarchy;
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());// CV_RETR_EXTERNAL
	if (!contours.empty() && !hierarchy.empty()) {
		vector<vector<Point> >::const_iterator itc = contours.begin();
		while (itc != contours.end()) {
			Rect rect = boundingRect(Mat(*itc));
			double area = contourArea(*itc);
			if (area < min_area) {

				for (int i = rect.y; i < rect.y + rect.height; i++) {  //ÈßÓà´úÂë£¬¿ÉÒÔÉ¾³ý
					uchar* output_data = dst.ptr<uchar>(i);
					for (int j = rect.x; j < rect.x + rect.width; j++) {
						if (output_data[j] == 1) {
								output_data[j] = 0;
						}
					}
				}

			}
			else {
				cout << "area = " << area << endl;
				Rect rect1;
				rect1.x = rect.x - m_padsize;
				rect1.y = rect.y - m_padsize;
				rect1.width = rect.width;
				rect1.height = rect.height;
				
				bboxs.push_back(rect1);

			}
			itc++;
		}
	}
}


vector<float> find_point(Mat& img, vector<Point>& points) {
	vector<float> p;
	for (int i = 0; i < points.size(); i++) {
		p.push_back(img.at<float>(points[i]));
	}
	return p;
}

vector<vector<float>>  nms(vector<vector<float>> &result){
	int l = result.size();
	float dis, maxscore;
	float x1_c,y1_c;
	float x2_c, y2_c;
	vector<vector<float>> temp;
	vector<float> maxs;
	vector<vector<float>> output;
	for(int i=0;i<l;i++){
		x1_c = (result[i][2] + result[i][4])/2.0;
		y1_c = (result[i][3] + result[i][5])/2.0;
		float score = result[i][1];
		for(int j= 0;j<l;j++){
			x2_c = (result[j][2] + result[j][4])/2.0;
			y2_c = (result[j][3] + result[j][5])/2.0;
			dis = sqrt((x1_c - x2_c)*(x1_c - x2_c) + (y1_c - y2_c)*(y1_c - y2_c));

			if(dis < 20){
				
				temp.push_back(result[j]);
			}
		}

		maxscore = temp[0][1];
		maxs = temp[0];
		for (int k = 0; k < temp.size(); ++k)
		{
			if(maxscore < temp[k][1]){
				maxscore = temp[k][1];
			}
		}

		if(maxscore == score || temp.size()==1){
			output.push_back(result[i]);
		}

		maxs.clear();
		temp.clear();

	}
	return output;
}

vector<string> getName(string namefile){
	vector<string> classNamesVec;
	ifstream classNamesFile(namefile);
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}
	return classNamesVec;
}

Scalar obj_id_to_color(int obj_id) {
    int const colors[6][3] = { { 1,0,1 },{ 0,1,1 }, { 0,0,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    int const offset = obj_id * 123457 % 6;
    int const color_scale = 150 + (obj_id * 123457) % 100;
    Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
    color *= color_scale;
    return color;
}


void draw(Mat &image_m, vector<vector<float>> result, string filepath, string filename){
	// result : [classid, confidence, xmin, ymin, xmax, ymax]
	Mat image = image_m.clone();
	vector<string> className = getName(filename);
	string cls;

	Scalar color;


	for (int i = 0; i < result.size(); ++i)
	{
		Point p(result[i][2], result[i][3]);
		string temp = to_string(result[i][1]);
		cls = className[result[i][0]] + ": " + temp.substr(0,5);
		color = obj_id_to_color(result[i][0]);

		rectangle(image, Rect(result[i][2], result[i][3], result[i][4] - result[i][2], result[i][5] - result[i][3]), color, 3);
		putText(image, cls, p, FONT_HERSHEY_PLAIN, 2, color, 3);

	}
	imwrite(filepath, image);

}


void saveTxt(string root, string filename,vector<vector<float>> result){
	// this function is written for save ouput
	// result 
	ofstream out(root + filename);
	string temp;
	for (int i = 0; i < result.size(); ++i)
	{
		temp = to_string(int(result[i][0]))+" "+to_string(int(result[i][2])) + " " + to_string(int(result[i][3]))+" " + to_string(int(result[i][4]-result[i][2]))+" "+ to_string(int(result[i][5] - result[i][3]));
		out<<temp.c_str()<<endl;
		// out.write(temp.c_str());
	}
	out.close();
}

bool is_target(Mat sub_img,int s, int object_id) {


	int thresh = getThreash(sub_img, 0.6);
	Mat binary_img;
	//sub_img.convertTo(temp, CV_8UC1);
	Mat sub_img1 = sub_img.clone();
	cvtColor(sub_img, sub_img, CV_BGR2GRAY);
	cvtColor(sub_img1, sub_img1, CV_BGR2GRAY);
	//threshold(sub_img1, binary_img, 0, 1, THRESH_BINARY + THRESH_OTSU);
	threshold(sub_img1, binary_img, thresh, 255, THRESH_BINARY);
	Scalar T1 = sum(sub_img.mul(binary_img / 255.0));
	Scalar T2 = sum(binary_img / 255.0);
	float f2 = T1[0] / max(T2[0],0.1);

	float a = max(sub_img.rows, sub_img.cols) / min(sub_img.rows, sub_img.cols);

	if (object_id == 0 && (sub_img.rows > 50 || sub_img.rows<10 || sub_img.cols > 50 || sub_img.cols<10 || a >4 || f2 <140)) { // 0 refers to car
		return false;
	}
	if (object_id == 1 && (sub_img.rows < 10 ||  sub_img.cols < 10 || f2 < 140 )) { // 1 refers to bridge
		return false;
	}


	/*imshow("X", binary_img);
	waitKey(0);*/
	
	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
	Mat New_img;
	
	morphologyEx(binary_img, New_img, 3, element);
	
	int num = findCounterNum(New_img);
	
	if (object_id==0 && num > s) {
		return false;
	}else{
		return true;
	}
	
}


bool is_car(vector<float> output, Mat &image){
		if(output[0] != 0){
			return false;
		}
	    int row = image.rows;
		int height = output[5] - output[3];
		if(height*5+output[3] < row){
			height = height*5;

		}
		else{
			height = row - output[3];
		}


		Mat target = image(Rect(output[2], output[3], output[4]-output[2], height)).clone();
		imshow("target", target);
		waitKey(100);
		return true;
}