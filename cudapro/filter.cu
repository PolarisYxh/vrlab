extern "C"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
#include <helper_cuda.h>
#include "Windows.h"
#include <math.h>
#include "mycudafunc.cuh"
#define STB_IMAGE_IMPLEMENTATION	// include֮ǰ���붨��
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION	// include֮ǰ���붨��
#include "stb_image_write.h"
using namespace std;

#define BLOCKDIM_X		16
#define BLOCKDIM_Y		16

#define GRIDDIM_X		256
#define GRIDDIM_Y		256
#define MASK_WIDTH		5

__constant__ int d_const_Gaussian[MASK_WIDTH * MASK_WIDTH]; //���䳣���洢��

static __global__ void kernel_GaussianFilt(int width, int height, int byteCount, unsigned char* d_src_imgbuf, unsigned char* d_guassian_imgbuf);
void GaussianFiltCpu(int width, int height, int byteCount, int Gaussian[][5], unsigned char* gray_imgbuf, unsigned char* guassian_imgbuf); //��˹�˲�
void GaussianFiltGpuMain()
{
	//printdevinfo();
	//�鿴�Կ�����
	struct cudaDeviceProp pror;
	cudaGetDeviceProperties(&pror, 0);
	cout << "maxThreadsPerBlock=" << pror.maxThreadsPerBlock << endl;

	long start, end;
	long time = 0;

	//CUDA��ʱ����
	start = GetTickCount();
	cudaEvent_t startt, stop; //CUDA��ʱ����
	cudaEventCreate(&startt);
	cudaEventCreate(&stop);
	cudaEventRecord(startt, 0);

	unsigned char* h_src_imgbuf;  //ͼ��ָ��
	int width, height, byteCount;
	char rootPath1[] = "nature_monte_gaussion.bmp";//"./nature_monte.bmp";
	char readPath[1024];
	int frame = 1;
		//sprintf(readPath, "%s%d.bmp", rootPath1, k);
		//h_src_imgbuf = readBmp(rootPath1, &width, &height, &byteCount);
		h_src_imgbuf = stbi_load(rootPath1, &width, &height, &byteCount, 0);
		int size1 = width * height * byteCount * sizeof(unsigned char);
		int size2 = width * height * sizeof(unsigned char);

		//���ͼ���ڴ�-host��	
		unsigned char* h_guassian_imgbuf = new unsigned char[width * height * byteCount];

		//�����Դ�ռ�
		unsigned char* d_src_imgbuf;
		unsigned char* d_guassian_imgbuf;

		cudaMalloc((void**)&d_src_imgbuf, size1);
		cudaMalloc((void**)&d_guassian_imgbuf, size1);

		//�����ݴ�Host����Device
		cudaMemcpy(d_src_imgbuf, h_src_imgbuf, size1, cudaMemcpyHostToDevice);

		//����˹ģ�崫��constant memory
		int Gaussian[25] = { 1,4,7,4,1,
							4,16,26,16,4,
							7,26,41,26,7,
							4,16,26,16,4,
							1,4,7,4,1 };//�ܺ�Ϊ273
		cudaMemcpyToSymbol(d_const_Gaussian, Gaussian, 25 * sizeof(int));

		int bx = ceil((double)width / BLOCKDIM_X); //����Ϳ�ķ���,ceil���ش��ڻ���ڱ��ʽ����С����
		int by = ceil((double)height / BLOCKDIM_Y);

		if (bx > GRIDDIM_X) bx = GRIDDIM_X;
		if (by > GRIDDIM_Y) by = GRIDDIM_Y;

		dim3 grid(bx, by);//����Ľṹ
		dim3 block(BLOCKDIM_X, BLOCKDIM_Y);//��Ľṹ

		//kernel--��˹�˲�
		kernel_GaussianFilt << <grid, block >> > (width, height, byteCount, d_src_imgbuf, d_guassian_imgbuf);
		cudaMemcpy(h_guassian_imgbuf, d_guassian_imgbuf, size1, cudaMemcpyDeviceToHost);//���ݴ���������

		char rootPath2[] = "./";
		char writePath[1024];
		//sprintf(writePath, "%s%d.bmp", rootPath2, k);
		char writepath[] = "./nature_monte_gaussionfilt.bmp";
		//saveBmp(writepath, h_guassian_imgbuf, width, height, byteCount);
		stbi_write_bmp(writepath, width, height, byteCount, h_guassian_imgbuf);
		//�������չʾ
		//cout << k << "  " << ((float)k / frame) * 100 << "%" << endl;

		//�ͷ��ڴ�
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
				int sum = 0;//��ʱֵ
				int tempPixelValue = 0;
				for (int m = -2; m <= 2; m++)
				{
					for (int n = -2; n <= 2; n++)
					{
						//�߽紦������Ԫ�ظ�ֵΪ��
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
	//��ʱ����
	long start, end;
	long time = 0;
	start = GetTickCount();

	unsigned char* src_imgbuf; //ͼ��ָ��
	int width, height, byteCount;
	char rootPath1[] = "nature_monte_gaussion.bmp";
	char readPath[1024];

	src_imgbuf = stbi_load(rootPath1, &width, &height, &byteCount, 0);
	//printf("��=%d����=%d���ֽ�=%d\n",width, height, byteCount);

	//�����˹ģ��ģ��
	int Gaussian_mask[5][5] = { {1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1} };//�ܺ�Ϊ273

	//���ͼ���ڴ����	
	unsigned char* guassian_imgbuf = new unsigned char[width * height * byteCount];

	//��ԭͼ��˹ģ��
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
	//��˹ģ������ 5��ѭ������
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < byteCount; k++)
			{
				int sum = 0;//��ʱֵ
				int tempPixelValue = 0;
				for (int m = -2; m <= 2; m++)
				{
					for (int n = -2; n <= 2; n++)
					{
						//�߽紦������Ԫ�ظ�ֵΪ��
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
