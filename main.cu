#include<iostream>
#include<stdio.h>

#include<opencv2/opencv.hpp>
#include<cublas_v2.h>
//Macro for checking cuda errors following a cuda launch or api call
void cudaCheckError()
{
	cudaError_t e=cudaGetLastError();
	if(e!=cudaSuccess)
	{
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
	}
}

//protoypes
void Low_Pass_Filter(float* input, float* output, int w, int h);
void GaussianBlur(float** input, float** output, int w, int h);
__global__ void Convolution(float*  input, float* output, float kernel[5], int w, int h);
__global__ void transpose_copy(float *odata, float* idata, int w, int h);

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

	//add padding for other operations
	cv::copyMakeBorder(right_image, right_image, 2, 2, 2, 2, cv::BORDER_CONSTANT, 0 );
	cv::copyMakeBorder(left_image, left_image, 2, 2, 2, 2, cv::BORDER_CONSTANT, 0 );

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

	//tests
	float* test = new float[w*h];
	float* test_gpu;
	cudaMalloc((void**) &test_gpu, sizeof(float)*w*h);
	//low pass filter
	std::cout << &right_image_data_gpu << std::endl;
	std::cout << &test_gpu << std::endl;
        GaussianBlur(&right_image_data_gpu, &test_gpu, w, h); 

	cudaMemcpy(test, test_gpu, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
	//print<<<1,1>>>(right_image_data_gpu, test,  w, h);
	//cudaDeviceSynchronize();
	cv::Mat A(h, w, CV_32FC1, test);
	cv::imshow("test", A);
	cv::waitKey(0);
	return 0;
}


/*
Input: 32*32
*/

__global__ void PipelinedBlur(float* input, float* output0, float* output1, float* output2, float* output3, int w)
{
	//assign shared memory to 32 by 32 with halo region
	__shared__ float shared_input[32][32];

	//assigned input to two dimensional as input is global
	//access 1D to 2D
	
	shared_input[threadIdx.y][threadIdx.x] = input[blockIdx.x*blockDim.x + ((threadIdx.y*w) + threadIdx.x)]; 

	// shared memory for blurred inputs
	__shared__ float blur0[32][32];
	__shared__ float blur1[32][32];
	__shared__ float blur2[32][32];
	__shared__ float blur3[32][32];
	__shared__ float blur4[32][32];

	//different convolution kernels with different sigmas
	__shared__ float sigma0[3][3] = {{0.077847, 0.123317, 0.077847}, {0.123317, 0.195346, 0.123317}, {0.077847, 0.123317, 0.077847}}; 
	__shared__ float sigma1[3][3] = {{0.102059, 0.115349, 0.102059}, {0.115349, 0.130371, 0.115349}, {0.102059, 0.115349, 0.102059}};
	__shared__ float sigma2[3][3] = {{0.107035, 0.113092, 0.107035}, {0.113092, 0.119491, 0.113092}, {0.107035, 0.113092, 0.107035}};
	__shared__ float sigma3[3][3] = {{0.108808, 0.112244, 0.108808}, {0.112244, 0.115788, 0.112244}, {0.108808, 0.112244, 0.108808}};
	__shared__ float sigma4[3][3] = {{0.109634, 0.111842, 0.109634}, {0.111842, 0.114094, 0.111842}, {0.109634, 0.111842, 0.109634}}; 
	
	//pixel
	//int pixel = threadIdx.x;
	
	//copy into shared memory
	shared_input[pixel] = input[pixel];

	//blur
	int sum0 = 0;
	int sum1 = 0; 
	int sum2 = 0; 
	int sum3 = 0; 
	int sum4 = 0; 
	for (int times = 0; times < 25; times++)
	{
		if ((pixel>=12)&&(pixel<(w-12)))
		{
			sum0 += sigma[times]*shared_input[times-pixel]
		}
	}
	blur1[pixel] = sigma0[0]*shared_input[pixel-12]


	//output differences
	__shared__ float difference0[];
	__shared__ float difference1[];
	__shared__ float difference2[];
	__shared__ float difference3[];
	*/

}




void GaussianBlur(float** input, float** output, int w, int h)
{
	float* kernel;
        float* stage1;
        cudaMallocManaged(&kernel, sizeof(float) * 5);
        cudaMallocManaged(&stage1, sizeof(float) * w * h);
	kernel[0] = 0.06136;
        kernel[1] = 0.24477;
	kernel[2] = 0.38774;
	kernel[3] = 0.24477;
	kernel[4] = 0.06136;
	Convolution<<<h, w>>>((*input), stage1, kernel, w, h);
	cudaCheckError();
	cudaDeviceSynchronize();
 	const float alf = 1;
	const float bet = 0;
 	const float *alpha = &alf;
	const float *beta = &bet;
        

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, h, w, alpha, stage1, w, beta, (*input), w, (*output), h); 
        //transpose_copy<<<dim3(w/32, h/32) , dim3(32, 8)>>>((*output), (*input), w , h);
	cudaCheckError();
        cudaDeviceSynchronize();
        Convolution<<<w, h>>>((*output), stage1, kernel, w, h);
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, w, h, alpha, stage1, h, beta, (*input), h, (*output), w);
}

__global__ void Convolution(float*  input, float* output, float kernel[5], int w, int h)
{
	//shared memory for faster accesses
	//__shared__ float buffer[pitch] = blockIdx.x
	//horizontal pass ... transpose and send thru again then transpose again
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (w)*(h); i += blockDim.x*gridDim.x)
	{
		output[i] = 0;
		if((threadIdx.x>=2)&&(threadIdx.x < 1002))
		{
			output[i] = input[i-2]*kernel[0] + input[i-1]*kernel[1] + input[i]*kernel[2] + input[i+1]*kernel[3] + input[i+2]*kernel[4];
			//printf("%f ", input[i]);
		}
	}
}


__global__ void transpose_copy(float *odata, float * idata, int w, int h)
{
  int TILE_DIM=32;
  int BLOCK_ROWS=8;
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS){
    if (x*width + (y + j) < w*h)   
    odata[x*width + (y+j)] = idata[(y+j)*width + x];



  }
}
