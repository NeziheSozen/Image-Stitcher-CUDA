#include<iostream>
#include<stdio.h>

#include<opencv2/opencv.hpp>

//protoypes
void Low_Pass_Filter(float* input, float* output, int w, int h);
__global__ void print(float *input, float* output, int w, int h);

int main(void)
{
	//basic parameters
	int w, h;
	float *right_image_data_gpu, *left_image_data_gpu;

	//pass through images
	cv::Mat right_image, left_image;
	
	//store original images in greyscale
	right_image = cv::imread("test.jpg", 0);
	left_image = cv::imread("test.jpg", 0);

	//pass width and height parameters and initialize array
	w = right_image.cols;
	h = right_image.rows;
	

	//display original images
	cv::imshow("Right Image", right_image);
	cv::waitKey(0);
	cv::imshow("Left Image", left_image);
	cv::waitKey(0);

	//convert to greyscale in float
	right_image.convertTo(right_image, CV_32FC1);
	left_image.convertTo(left_image, CV_32FC1);
	right_image /= 255;
	left_image /= 255;

	float* right_image_data = (float*)right_image.data;
	float* left_image_data = (float*)left_image.data;

	//data allocation and copy to gpu
	cudaMalloc((void**)&right_image_data_gpu, sizeof(float)*h*w);
	cudaMemcpy(right_image_data_gpu, right_image_data, sizeof(float)*h*w, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&left_image_data_gpu, sizeof(float)*h*w);
	cudaMemcpy(left_image_data_gpu, left_image_data, sizeof(float)*h*w, cudaMemcpyHostToDevice);

	//low pass filter

	
	//tests
	float* test = new float[w*h];
	cudaMemcpy(test, right_image_data_gpu, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
	//print<<<1,1>>>(right_image_data_gpu, test,  w, h);
	//cudaDeviceSynchronize();
	cv::Mat A(h, w, CV_32FC1, test);
	cv::imshow("test", A);
	cv::waitKey(0);
	return 0;
}


__global__ void print(float *input, float* output, int w, int h)
{
	for(int i = 0; i<w*h; i++)
	{
		output[i] = input[i];
	}
}



/*
__global__ Convolution(float* input, float* output, float* kernel, int pitch, int w, int h)
{
	//shared memory for faster accesses
	//__shared__ float buffer[pitch] = blockIdx.x

	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < w*h; i += blockDim.x*gridDim.x)
	{
		//possible divergence
		int arrayVal_back = i>=0 ? input[i] : 0;
		int arrayVal_back = i<(h*w) ? input[i] : 0;
		int arrayVal = i>=0 ? input[i] : 0;
		int arrayVal = i>=0 ? input[i] : 0;
		int arrayVal = i<(h*w) ? input[i] : 0;
		int arrayVal = i<(h*w) ? input[i] : 0;
		if ((i>=0) || (i<(h*w)))
		{
			output[i] = arrayVal*kernel[0] + arrayVal
		}
	}
}

// low pass filter implementation
void Low_Pass_Filter(float* input, float* output, int w, int h)
{
	
}
*/
