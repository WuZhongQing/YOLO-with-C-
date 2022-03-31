#pragma once
// #include<io.h>
#include<iostream>
#include <sys/io.h>
#include<vector>
#include<fstream>
#include"opencv2/opencv.hpp"
#include <dirent.h>
#include<string>




using namespace std;
using namespace cv;





// void getFiles(string path, vector<string>& files);
vector<string> getFiles(string root);

vector<double>  im2SIFT(const Mat& img);

void readdatas(vector<vector<double>>& datav, bool flag, string& model); 

//ÏÂÃæÊÇÍ¼Æ¬µÄ²Ù×÷
void find(Mat& image, float num, vector<Point>& points);
float m_mean(Mat& img, vector<Point>& points);
float m_std(Mat& img, vector<Point>& points, float miu);
void bwareaopen(Mat& src, Mat& dst, double min_area, vector<Rect>& bboxs, int m_padsize);
vector<float> find_point(Mat& img, vector<Point>& points);

//ÏÂÃæÊÇ¶Ôbbox²Ù×÷µÄº¯Êý
int findCounterNum(Mat& img);
int getThreash(const Mat &img, double a);


vector<vector<float>>  nms(vector<vector<float>> &result);
void draw(Mat &image, vector<vector<float>> result, string filepath, string filename);
vector<string> getName(string namefile);
Scalar obj_id_to_color(int obj_id);
void saveTxt(string root, string filename, vector<vector<float>> result);
bool is_target(Mat sub_img,int s, int object_id);

bool is_car(vector<float> output, Mat &image);
// vector<vector<float>> nms(vector<vector<float>>);