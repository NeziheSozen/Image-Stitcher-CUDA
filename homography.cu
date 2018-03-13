#include<iostream>
#include<stdio.h>


#include <opencv2/features2d.hpp>
#include<opencv2/opencv.hpp>

int main(void)
{
	//basic parameters
	int w, h;
	float *right_image_data_gpu, *left_image_data_gpu;

        //pass through images
	cv::Mat right_image, left_image;

        //store original images in greyscale
	right_image = cv::imread("J4.jpg", 0);
	left_image = cv::imread("J3.jpg", 0);
	//Cv::FAST variables
	int threshold=9;
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        //cv::FAST(right_image,keypoints_1,threshold,true);
	//cv::FAST(left_image, keypoints_2,threshold,true);

	
	// Add results to image and save.

	cv::Mat descriptors_1, descriptors_2;
	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
	detector->detect(right_image, keypoints_1);
	detector->detect(left_image, keypoints_2);

        cv::Mat output;
        cv::drawKeypoints(right_image, keypoints_1, output);
        cv::imshow("QTIE", output);
        cv::waitKey(0);





	cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
	extractor->compute( right_image, keypoints_1, descriptors_1 );
	extractor->compute( left_image, keypoints_2, descriptors_2 );
	
if ( descriptors_1.empty() )
   cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
if ( descriptors_2.empty() )
   cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

        descriptors_1.convertTo(descriptors_1, CV_32F);
        descriptors_2.convertTo(descriptors_2, CV_32F);
	//-- Step 2: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );
	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_1.rows; i++ )
  	{
 	  double dist = matches[i].distance;
    	  if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< cv::DMatch > good_matches;
	for( int i = 0; i < descriptors_1.rows; i++ )
  	{
		if( matches[i].distance <= max(2*min_dist, 0.02) )
		{
			good_matches.push_back( matches[i]);
		}
	}
	//-- Draw only "good" matches
	cv::Mat img_matches;
	cv::drawMatches( right_image, keypoints_1, left_image, keypoints_2, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//-- Show detected matches
	cv::imshow( "Good Matches", img_matches );
	for( int i = 0; i < (int)good_matches.size(); i++ )
	{
		printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
	}
	cv::waitKey(0);
	return 0;
}
