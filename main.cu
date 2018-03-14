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
__global__ void PipelinedBlur(float* input, int w);

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

	//cudaMalloc((void**)&left_image_data_gpu, sizeof(float)*h*w);
	//cudaMemcpy(left_image_data_gpu, left_image_data, sizeof(float)*h*w, cudaMemcpyHostToDevice);

	//tests
	float* test = new float[w*h];
	float* test_gpu;
	cudaMalloc((void**) &test_gpu, sizeof(float)*w*h);
	//low pass filter
	std::cout << &right_image_data_gpu << std::endl;
	std::cout << &test_gpu << std::endl;
        GaussianBlur(&right_image_data_gpu, &test_gpu, w, h); 
	
	int blockSize = w/32;
        PipelinedBlur<<< blockSize ,dim3(32,32) >>>(right_image_data, w);
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

__global__ void PipelinedBlur(float* input, int w)
{
	float INTENSITY_THRESHOLD = 3.5f;
	float CORNER_THRESHOLD = 0.0f;
	
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
	float sigma0[3][3] = {{0.077847, 0.123317, 0.077847}, {0.123317, 0.195346, 0.123317}, {0.077847, 0.123317, 0.077847}}; 
	float sigma1[3][3] = {{0.102059, 0.115349, 0.102059}, {0.115349, 0.130371, 0.115349}, {0.102059, 0.115349, 0.102059}};
	float sigma2[3][3] = {{0.107035, 0.113092, 0.107035}, {0.113092, 0.119491, 0.113092}, {0.107035, 0.113092, 0.107035}};
	float sigma3[3][3] = {{0.108808, 0.112244, 0.108808}, {0.112244, 0.115788, 0.112244}, {0.108808, 0.112244, 0.108808}};
	float sigma4[3][3] = {{0.109634, 0.111842, 0.109634}, {0.111842, 0.114094, 0.111842}, {0.109634, 0.111842, 0.109634}};
	//output differences
	__shared__ float difference0[32][32];
	__shared__ float difference1[32][32];
	__shared__ float difference2[32][32];
	__shared__ float difference3[32][32];

	//pixel
	int x = threadIdx.x;
	int y = threadIdx.y;/*
	if (x == 0 && y == 0)
          sigma0[0][0] = 0.077847;
        else if (x == 0 && y == 1)
          sigma0[0][1] = 0.123317;
        else if (x == 0 && y == 2)
          sigma[0][2] = 0.077847;
	else if (x == 1 && y == 0)
*/
	if (((x>0)&&(y>0))&&((x<31)&&(y<31)))
	{
		//convolutions
		blur0[y][x]
		= (sigma0[0][0]*shared_input[y-1][x-1]) + (sigma0[0][1]*shared_input[y-1][x]) + (sigma0[0][2]*shared_input[y-1][x+1])
		+ (sigma0[1][0]*shared_input[y][x-1]) + (sigma0[1][1]*shared_input[y][x]) + (sigma0[1][2]*shared_input[y][x+1])
		+ (sigma0[2][0]*shared_input[y+1][x-1]) + (sigma0[2][1]*shared_input[y+1][x]) + (sigma0[2][2]*shared_input[y+1][x+1]);


		blur1[y][x]
		= (sigma1[0][0]*shared_input[y-1][x-1]) + (sigma1[0][1]*shared_input[y-1][x]) + (sigma1[0][2]*shared_input[y-1][x+1])
		+ (sigma1[1][0]*shared_input[y][x-1]) + (sigma1[1][1]*shared_input[y][x]) + (sigma1[1][2]*shared_input[y][x+1])
		+ (sigma1[2][0]*shared_input[y+1][x-1]) + (sigma1[2][1]*shared_input[y+1][x]) + (sigma1[2][2]*shared_input[y+1][x+1]);


		blur2[y][x]
		= (sigma2[0][0]*shared_input[y-1][x-1]) + (sigma2[0][1]*shared_input[y-1][x]) + (sigma2[0][2]*shared_input[y-1][x+1])
		+ (sigma2[1][0]*shared_input[y][x-1]) + (sigma2[1][1]*shared_input[y][x]) + (sigma2[1][2]*shared_input[y][x+1])
		+ (sigma2[2][0]*shared_input[y+1][x-1]) + (sigma2[2][1]*shared_input[y+1][x]) + (sigma2[2][2]*shared_input[y+1][x+1]);

		blur3[y][x]
		= (sigma3[0][0]*shared_input[y-1][x-1]) + (sigma3[0][1]*shared_input[y-1][x]) + (sigma3[0][2]*shared_input[y-1][x+1])
		+ (sigma3[1][0]*shared_input[y][x-1]) + (sigma3[1][1]*shared_input[y][x]) + (sigma3[1][2]*shared_input[y][x+1])
		+ (sigma3[2][0]*shared_input[y+1][x-1]) + (sigma3[2][1]*shared_input[y+1][x]) + (sigma3[2][2]*shared_input[y+1][x+1]);

		blur4[y][x]
		= (sigma4[0][0]*shared_input[y-1][x-1]) + (sigma4[0][1]*shared_input[y-1][x]) + (sigma4[0][2]*shared_input[y-1][x+1])
		+ (sigma4[1][0]*shared_input[y][x-1]) + (sigma4[1][1]*shared_input[y][x]) + (sigma4[1][2]*shared_input[y][x+1])
		+ (sigma4[2][0]*shared_input[y+1][x-1]) + (sigma4[2][1]*shared_input[y+1][x]) + (sigma4[2][2]*shared_input[y+1][x+1]);
	}
	__syncthreads();

	//difference of gaussian
	difference0[y][x] = blur0[y][x] - blur1[y][x];
	difference1[y][x] = blur1[y][x] - blur2[y][x];
	difference2[y][x] = blur2[y][x] - blur3[y][x];
	difference3[y][x] = blur3[y][x] - blur4[y][x];
	__syncthreads();

	//find neighbors in 3x3 cube 
	if (((x>0)&&(y>0))&&((x<31)&&(y<31)))
	{
		//could have used reduction with shared memory but we want to keep more space for shared memory
		
		//upper min calc
		float upper0_123 = fminf(fminf(difference0[y-1][x-1], difference0[y-1][x-1]), difference0[y-1][x+1]);
		float upper0_456 = fminf(fminf(difference0[y][x-1], difference0[y][x-1]), difference0[y][x+1]);
		float upper0_789 = fminf(fminf(difference0[y+1][x-1], difference0[y+1][x-1]), difference0[y+1][x+1]);
		
		float upper1_123 = fminf(fminf(difference1[y-1][x-1], difference1[y-1][x-1]), difference1[y-1][x+1]);
		float upper1_456 = fminf(fminf(difference1[y][x-1], difference1[y][x-1]), difference1[y][x+1]);
		float upper1_789 = fminf(fminf(difference1[y+1][x-1], difference1[y+1][x-1]), difference1[y+1][x+1]);


		float upper2_123 = fminf(fminf(difference2[y-1][x-1], difference2[y-1][x-1]), difference2[y-1][x+1]);
		float upper2_456 = fminf(fminf(difference2[y][x-1], difference2[y][x-1]), difference2[y][x+1]);
		float upper2_789 = fminf(fminf(difference2[y+1][x-1], difference2[y+1][x-1]), difference2[y+1][x+1]);

		float upper0 = fminf(fminf(upper0_123, upper0_456), upper0_789);
		float upper1 = fminf(fminf(upper1_123, upper1_456), upper1_789);
		float upper2 = fminf(fminf(upper2_123, upper2_456), upper2_789);

		float upper_final_min = fminf(fminf(upper0, upper1), upper2);

		//upper max calc
		upper0_123 = fmaxf(fmaxf(difference0[y-1][x-1], difference0[y-1][x-1]), difference0[y-1][x+1]);
		upper0_456 = fmaxf(fmaxf(difference0[y][x-1], difference0[y][x-1]), difference0[y][x+1]);
		upper0_789 = fmaxf(fmaxf(difference0[y+1][x-1], difference0[y+1][x-1]), difference0[y+1][x+1]);
		
		upper1_123 = fmaxf(fmaxf(difference1[y-1][x-1], difference1[y-1][x-1]), difference1[y-1][x+1]);
		upper1_456 = fmaxf(fmaxf(difference1[y][x-1], difference1[y][x-1]), difference1[y][x+1]);
		upper1_789 = fmaxf(fmaxf(difference1[y+1][x-1], difference1[y+1][x-1]), difference1[y+1][x+1]);


		upper2_123 = fmaxf(fmaxf(difference2[y-1][x-1], difference2[y-1][x-1]), difference2[y-1][x+1]);
		upper2_456 = fmaxf(fmaxf(difference2[y][x-1], difference2[y][x-1]), difference2[y][x+1]);
		upper2_789 = fmaxf(fmaxf(difference2[y+1][x-1], difference2[y+1][x-1]), difference2[y+1][x+1]);

		upper0 = fmaxf(fmaxf(upper0_123, upper0_456), upper0_789);
		upper1 = fmaxf(fmaxf(upper1_123, upper1_456), upper1_789);
		upper2 = fmaxf(fmaxf(upper2_123, upper2_456), upper2_789);

		float upper_final_max = fmaxf(fmaxf(upper0, upper1), upper2);

		//lower min calc
		float lower3_123 = fminf(fminf(difference3[y-1][x-1], difference3[y-1][x-1]), difference3[y-1][x+1]);
		float lower3_456 = fminf(fminf(difference3[y][x-1], difference3[y][x-1]), difference3[y][x+1]);
		float lower3_789 = fminf(fminf(difference3[y+1][x-1], difference3[y+1][x-1]), difference3[y+1][x+1]);
		
		float lower1_123 = fminf(fminf(difference1[y-1][x-1], difference1[y-1][x-1]), difference1[y-1][x+1]);
		float lower1_456 = fminf(fminf(difference1[y][x-1], difference1[y][x-1]), difference1[y][x+1]);
		float lower1_789 = fminf(fminf(difference1[y+1][x-1], difference1[y+1][x-1]), difference1[y+1][x+1]);


		float lower2_123 = fminf(fminf(difference2[y-1][x-1], difference2[y-1][x-1]), difference2[y-1][x+1]);
		float lower2_456 = fminf(fminf(difference2[y][x-1], difference2[y][x-1]), difference2[y][x+1]);
		float lower2_789 = fminf(fminf(difference2[y+1][x-1], difference2[y+1][x-1]), difference2[y+1][x+1]);

		float lower3 = fminf(fminf(lower3_123, lower3_456), lower3_789);
		float lower1 = fminf(fminf(lower1_123, lower1_456), lower1_789);
		float lower2 = fminf(fminf(lower2_123, lower2_456), lower2_789);

		float lower_final_min = fminf(fminf(lower3, lower1), lower2);

		//upper max calc
		lower3_123 = fmaxf(fmaxf(difference3[y-1][x-1], difference3[y-1][x-1]), difference3[y-1][x+1]);
		lower3_456 = fmaxf(fmaxf(difference3[y][x-1], difference3[y][x-1]), difference3[y][x+1]);
		lower3_789 = fmaxf(fmaxf(difference3[y+1][x-1], difference3[y+1][x-1]), difference3[y+1][x+1]);
		
		lower1_123 = fmaxf(fmaxf(difference1[y-1][x-1], difference1[y-1][x-1]), difference1[y-1][x+1]);
		lower1_456 = fmaxf(fmaxf(difference1[y][x-1], difference1[y][x-1]), difference1[y][x+1]);
		lower1_789 = fmaxf(fmaxf(difference1[y+1][x-1], difference1[y+1][x-1]), difference1[y+1][x+1]);


		lower2_123 = fmaxf(fmaxf(difference2[y-1][x-1], difference2[y-1][x-1]), difference2[y-1][x+1]);
		lower2_456 = fmaxf(fmaxf(difference2[y][x-1], difference2[y][x-1]), difference2[y][x+1]);
		lower2_789 = fmaxf(fmaxf(difference2[y+1][x-1], difference2[y+1][x-1]), difference2[y+1][x+1]);

		lower3 = fmaxf(fmaxf(lower3_123, lower3_456), lower3_789);
		lower1 = fmaxf(fmaxf(lower1_123, lower1_456), lower1_789);
		lower2 = fmaxf(fmaxf(lower2_123, lower2_456), lower2_789);

		float lower_final_max = fmaxf(fmaxf(lower3, lower1), lower2);
		
		//generate extrema images
		//difference1 is upper
		//difference2 is lower
		//think of it visually
		if ((upper_final_max == difference1[y][x]) || (upper_final_min == difference1[y][x]) || (lower_final_max == difference2[y][x]) || (lower_final_min == difference2[y][x]))
		{
			//subpixel value not calculated so may not be robust and give excat....
			
			//intensity threshold
			if (((difference1[y][x] > INTENSITY_THRESHOLD) && (difference1[y][x] < INTENSITY_THRESHOLD)) || ((difference2[y][x] > INTENSITY_THRESHOLD) && (difference2[y][x] < INTENSITY_THRESHOLD)))
			{
				//compute hessian matrix
				//assume h^2 = k^2 = 1024
				float difference1xx = (1.0/1024)*(difference1[y][x+1]-(2*difference1[y][x])-difference1[y][x-1]);
				float difference1yy = (1.0/1024)*(difference1[y+1][x]-(2*difference1[y][x])-difference1[y-1][x]);
				float difference1xy = (1.0/(4*1024))*(difference1[y+1][x+1]-difference1[y-1][x+1]-difference1[y+1][x-1] + difference1[y-1][x-1]);

				float difference2xx = (1.0/1024)*(difference2[y][x+1]-(2*difference2[y][x])-difference2[y][x-1]);
				float difference2yy = (1.0/1024)*(difference2[y+1][x]-(2*difference2[y][x])-difference2[y-1][x]);
				float difference2xy = (1.0/(4*1024))*(difference2[y+1][x+1]-difference2[y-1][x+1]-difference2[y+1][x-1] + difference2[y-1][x-1]);

				//find ratio of eigenvalues
				float eigen1_positive = (1.0/2.0)*(difference1xx + difference1yy + sqrtf(((difference1xx-difference1yy)*(difference1xx-difference1yy))+(4.0*difference1xy*difference1xy)));
				float eigen1_negative = (1.0/2.0)*(difference1xx + difference1yy - sqrtf(((difference1xx-difference1yy)*(difference1xx-difference1yy))+(4.0*difference1xy*difference1xy)));
		
				float eigen2_positive = (1.0/2.0)*(difference2xx + difference2yy + sqrtf(((difference2xx-difference2yy)*(difference2xx-difference2yy))+(4.0*difference2xy*difference2xy)));
				float eigen2_negative = (1.0/2.0)*(difference2xx + difference2yy - sqrtf(((difference2xx-difference2yy)*(difference2xx-difference2yy))+(4.0*difference2xy*difference2xy)));

				//shi-tomasi ?? score
				float score1 = fminf(eigen1_negative, eigen1_positive);
				float score2 = fminf(eigen2_negative, eigen2_positive);

				//acceptable corners
				if ((score1 > CORNER_THRESHOLD) || (score2 > CORNER_THRESHOLD))
				{
					//implement orientation if time permits
					printf("X: %d, Y: %d, Data: %d", x, y, shared_input[y][x]);
					//return the keypoints in __shared__ and plot them over the original photo
				}
			}
		}
		
	}
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
