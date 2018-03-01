//g++ -o test_1 test_1.cpp `pkg-config opencv --cflags --libs` 
#include <iostream>  
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//main function for testing and stuff for now
int main()
{
	//open photos using opencv
	cv::Mat limg, rimg;
	cv::imread("picture1.jpg", 0).convertTo(limg, CV_32FC1);
//	cv::imread("data/righ.pgm", 0).convertTo(rimg, CV_32FC1);
	unsigned int w = limg.cols;
	unsigned int h = limg.rows;
	std::cout << "Image size = (" << w << "," << h << ")" << std::endl;
	return 0;
}
