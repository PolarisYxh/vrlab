extern "C"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
#include <helper_cuda.h>
#include "Windows.h"
#include <math.h>
#include "mycudafunc.cuh"
#define STB_IMAGE_IMPLEMENTATION	// include之前必须定义
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION	// include之前必须定义
#include "stb_image_write.h"
using namespace std;

#define BLOCKDIM_X		16
#define BLOCKDIM_Y		16

#define GRIDDIM_X		256
#define GRIDDIM_Y		256
#define MASK_WIDTH		5

__constant__ int d_const_Gaussian[MASK_WIDTH * MASK_WIDTH]; //分配常数存储器

static __global__ void kernel_GaussianFilt(int width, int height, int byteCount, unsigned char* d_src_imgbuf, unsigned char* d_guassian_imgbuf);
void GaussianFiltCpu(int width, int height, int byteCount, int Gaussian[][5], unsigned char* gray_imgbuf, unsigned char* guassian_imgbuf); //高斯滤波
void GaussianFiltGpuMain()
{
	//printdevinfo();
	//查看显卡配置
	struct cudaDeviceProp pror;
	cudaGetDeviceProperties(&pror, 0);
	cout << "maxThreadsPerBlock=" << pror.maxThreadsPerBlock << endl;

	long start, end;
	long time = 0;

	//CUDA计时函数
	start = GetTickCount();
	cudaEvent_t startt, stop; //CUDA计时机制
	cudaEventCreate(&startt);
	cudaEventCreate(&stop);
	cudaEventRecord(startt, 0);

	unsigned char* h_src_imgbuf;  //图像指针
	int width, height, byteCount;
	char rootPath1[] = "nature_monte_gaussion.bmp";//"./nature_monte.bmp";
	char readPath[1024];
	int frame = 1;
		//sprintf(readPath, "%s%d.bmp", rootPath1, k);
		//h_src_imgbuf = readBmp(rootPath1, &width, &height, &byteCount);
		h_src_imgbuf = stbi_load(rootPath1, &width, &height, &byteCount, 0);
		int size1 = width * height * byteCount * sizeof(unsigned char);
		int size2 = width * height * sizeof(unsigned char);

		//输出图像内存-host端	
		unsigned char* h_guassian_imgbuf = new unsigned char[width * height * byteCount];

		//分配显存空间
		unsigned char* d_src_imgbuf;
		unsigned char* d_guassian_imgbuf;

		cudaMalloc((void**)&d_src_imgbuf, size1);
		cudaMalloc((void**)&d_guassian_imgbuf, size1);

		//把数据从Host传到Device
		cudaMemcpy(d_src_imgbuf, h_src_imgbuf, size1, cudaMemcpyHostToDevice);

		//将高斯模板传入constant memory
		int Gaussian[25] = { 1,4,7,4,1,
							4,16,26,16,4,
							7,26,41,26,7,
							4,16,26,16,4,
							1,4,7,4,1 };//总和为273
		cudaMemcpyToSymbol(d_const_Gaussian, Gaussian, 25 * sizeof(int));

		int bx = ceil((double)width / BLOCKDIM_X); //网格和块的分配,ceil返回大于或等于表达式的最小整数
		int by = ceil((double)height / BLOCKDIM_Y);

		if (bx > GRIDDIM_X) bx = GRIDDIM_X;
		if (by > GRIDDIM_Y) by = GRIDDIM_Y;

		dim3 grid(bx, by);//网格的结构
		dim3 block(BLOCKDIM_X, BLOCKDIM_Y);//块的结构

		//kernel--高斯滤波
		kernel_GaussianFilt << <grid, block >> > (width, height, byteCount, d_src_imgbuf, d_guassian_imgbuf);
		cudaMemcpy(h_guassian_imgbuf, d_guassian_imgbuf, size1, cudaMemcpyDeviceToHost);//数据传回主机端

		char rootPath2[] = "./";
		char writePath[1024];
		//sprintf(writePath, "%s%d.bmp", rootPath2, k);
		char writepath[] = "./nature_monte_gaussionfilt.bmp";
		//saveBmp(writepath, h_guassian_imgbuf, width, height, byteCount);
		stbi_write_bmp(writepath, width, height, byteCount, h_guassian_imgbuf);
		//输出进度展示
		//cout << k << "  " << ((float)k / frame) * 100 << "%" << endl;

		//释放内存
		cudaFree(d_src_imgbuf);
		cudaFree(d_guassian_imgbuf);

		stbi_image_free(h_src_imgbuf);
		delete[]h_guassian_imgbuf;
	end = GetTickCount();
	InterlockedExchangeAdd(&time, end - start);
	cout << "Total time GPU:";
	cout << time << endl;
	int x;
	cin >> x;
}

static __global__ void kernel_GaussianFilt(int width, int height, int byteCount, unsigned char* d_src_imgbuf, unsigned char* d_dst_imgbuf)
{
	const int tix = blockDim.x * blockIdx.x + threadIdx.x;
	const int tiy = blockDim.y * blockIdx.y + threadIdx.y;

	const int threadTotalX = blockDim.x * gridDim.x;
	const int threadTotalY = blockDim.y * gridDim.y;

	for (int ix = tix; ix < height; ix += threadTotalX)
		for (int iy = tiy; iy < width; iy += threadTotalY)
		{
			for (int k = 0; k < byteCount; k++)
			{
				int sum = 0;//临时值
				int tempPixelValue = 0;
				for (int m = -2; m <= 2; m++)
				{
					for (int n = -2; n <= 2; n++)
					{
						//边界处理，幽灵元素赋值为零
						if (ix + m < 0 || iy + n < 0 || ix + m >= height || iy + n >= width)
							tempPixelValue = 0;
						else
							tempPixelValue = *(d_src_imgbuf + (ix + m) * width * byteCount + (iy + n) * byteCount + k);
						sum += tempPixelValue * d_const_Gaussian[(m + 2) * 5 + n + 2];
					}
				}

				if (sum / 273 < 0)
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = 0;
				else if (sum / 273 > 255)
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = 255;
				else
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = sum / 273;
			}
		}
}
void main1()
{
	//计时函数
	long start, end;
	long time = 0;
	start = GetTickCount();

	unsigned char* src_imgbuf; //图像指针
	int width, height, byteCount;
	char rootPath1[] = "nature_monte_gaussion.bmp";
	char readPath[1024];

	src_imgbuf = stbi_load(rootPath1, &width, &height, &byteCount, 0);
	//printf("宽=%d，高=%d，字节=%d\n",width, height, byteCount);

	//读入高斯模糊模板
	int Gaussian_mask[5][5] = { {1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1} };//总和为273

	//输出图像内存分配	
	unsigned char* guassian_imgbuf = new unsigned char[width * height * byteCount];

	//对原图高斯模糊
	GaussianFiltCpu(width, height, byteCount, Gaussian_mask, src_imgbuf, guassian_imgbuf);

	char rootPath2[] = "./";
	char writePath[] = "nature_monte_gaussion_cpufilt.bmp";
	stbi_write_bmp(writePath, width, height, byteCount, guassian_imgbuf);
	delete[]src_imgbuf;
	delete[]guassian_imgbuf;
	end = GetTickCount();
	InterlockedExchangeAdd(&time, end - start);
	cout << "Total time CPU:";
	cout << time << endl;
	int x;
	cin >> x;
}

void GaussianFiltCpu(int width, int height, int byteCount, int Gaussian[][5], unsigned char* src_imgbuf, unsigned char* guassian_imgbuf)
{
	//高斯模糊处理 5层循环处理
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < byteCount; k++)
			{
				int sum = 0;//临时值
				int tempPixelValue = 0;
				for (int m = -2; m <= 2; m++)
				{
					for (int n = -2; n <= 2; n++)
					{
						//边界处理，幽灵元素赋值为零
						if (i + m < 0 || j + n < 0 || i + m >= height || j + n >= width)
							tempPixelValue = 0;
						else
							tempPixelValue = *(src_imgbuf + (i + m) * width * byteCount + (j + n) * byteCount + k);
						//tempPixelValue=*(gray_imgbuf+(i+m)*width+(j+n)+k);	
						sum += tempPixelValue * Gaussian[m + 2][n + 2];
					}
				}
				//tempPixelValue=*(src_imgbuf+(i)*width*byteCount+(j)*byteCount+k);
				if (sum / 273 < 0)
					*(guassian_imgbuf + i * width * byteCount + j * byteCount + k) = 0;
				else if (sum / 273 > 255)
					*(guassian_imgbuf + i * width * byteCount + j * byteCount + k) = 255;
				else
					*(guassian_imgbuf + i * width * byteCount + j * byteCount + k) = sum / 273;
			}
		}
	}
}
